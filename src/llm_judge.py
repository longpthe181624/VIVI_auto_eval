"""
Single-Call LLM Judge Module.

For test cases the keyword-overlap similarity scorer (eval_tools.compute_similarity)
cannot confidently resolve (the RETEST / borderline band), this module escalates to
a single local LLM call that judges whether the bot's answer is correct/acceptable
against the reference text, even when phrased differently from it.

A 2-3 call multi-agent debate design (Advocate vs Skeptic, optionally + a Judge
synthesis) was tried first and discarded: with the small local models available
here (qwen2.5:3b, llama3.2:latest running on CPU), the "Advocate" role - tasked
with defending the answer even when it should not be defensible - reliably talked
the Judge into false PASS verdicts on cases a plain single judge call correctly
failed (e.g. generic canned-greeting non-answers, visibly truncated answers).
Manual spot-checks found the single-call judge more accurate AND ~5-6x faster
(no adversarial framing to be misled by, and 1 model call instead of 2-3).

Runs entirely on a local Ollama model (no OpenAI / paid API calls). Requires the
Ollama service running locally (`ollama serve`) with the model below pulled.
"""

import re
import time
import logging
import threading
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# Was temporarily switched to qwen2.5:1.5b while Ollama was misconfigured to
# run inference through a software Vulkan (llvmpipe) fallback instead of its
# native CPU backend, making every call extremely slow. That root cause is
# now fixed (`OLLAMA_LLM_LIBRARY=cpu`), and a real-data run showed the 1.5b
# model got 58% of its own PASS verdicts wrong - specifically, it kept
# rationalizing clearly-wrong decline messages ("Vui lòng kết nối Internet để
# sử dụng tính năng này") as acceptable, the same failure mode the discarded
# multi-agent debate design had. With the CPU backend fixed, qwen2.5:3b runs
# fast enough (~2-10s/call) and correctly judges those same cases as FAIL, so
# there's no longer a reason to trade accuracy for speed here.
JUDGE_MODEL = "qwen2.5:3b"

REQUEST_TIMEOUT_SEC = 60
NUM_PREDICT = 30

# The local Ollama model server runs with a single processing slot (`-np 1`) -
# it can only handle one generation request at a time no matter how many
# threads call it concurrently. When the batch evaluator runs many rows in
# parallel (16 workers) and several land on RETEST in the same window, firing
# all of their judge calls at once just makes them pile up in Ollama's queue;
# by the time a request further back in the queue is served, the client-side
# REQUEST_TIMEOUT_SEC has often already elapsed, so it fails instead of
# waiting. This semaphore caps concurrent judge calls to match what the
# backend can actually run, so requests queue up cleanly client-side (each
# still gets its own full timeout budget once it's actually its turn) instead
# of stacking up and timing out together.
_OLLAMA_CONCURRENCY = threading.Semaphore(1)

# Circuit breaker: the semaphore above only fixes *queueing* pileup. It does not
# help when the host machine itself is so CPU-starved that Ollama cannot serve
# even a single trivial request within the timeout (observed: 90s+ for a
# 10-token completion under heavy load). In that state, every subsequent judge
# call is doomed to wait the full REQUEST_TIMEOUT_SEC before failing - on a
# large batch that adds up to many wasted minutes. After a few consecutive
# failures, stop attempting further judge calls for a cooldown period so rows
# fall straight back to their keyword-scored RETEST verdict instantly instead
# of waiting out a timeout each time. Mirrors the BatchHealthMonitor pattern
# already used for web search in eval_tools.py.
_CONSECUTIVE_FAILURE_LIMIT = 3
_CIRCUIT_COOLDOWN_SEC = 120

_circuit_lock = threading.Lock()
_consecutive_failures = 0
_circuit_open_until = 0.0

_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)


def _circuit_is_open() -> bool:
    with _circuit_lock:
        return time.time() < _circuit_open_until


def _record_outcome(success: bool) -> None:
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        if success:
            _consecutive_failures = 0
            _circuit_open_until = 0.0
            return
        _consecutive_failures += 1
        if _consecutive_failures >= _CONSECUTIVE_FAILURE_LIMIT:
            _circuit_open_until = time.time() + _CIRCUIT_COOLDOWN_SEC
            logger.error(
                f"LLM judge circuit breaker OPEN: {_consecutive_failures} consecutive failures. "
                f"Skipping judge calls for {_CIRCUIT_COOLDOWN_SEC}s - Ollama appears unresponsive "
                f"(check system load / `ollama serve` health)."
            )


def reset_circuit() -> None:
    """Manually clears the circuit breaker state (mainly for tests)."""
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        _consecutive_failures = 0
        _circuit_open_until = 0.0


def _build_judge_prompt(user_cmd: str, actual_resp: str, expected_ref: str) -> str:
    return f"""Bạn là giám khảo đánh giá chatbot xe VinFast. Đọc câu hỏi, câu trả lời của bot, và tài liệu tham khảo.
Phán quyết câu trả lời có ĐÚNG/CHẤP NHẬN ĐƯỢC không (kể cả khi diễn giải lại theo ý, không cần lặp nguyên văn).
Nếu câu trả lời là câu chào chung chung, lạc đề, hoặc bị cụt/không đầy đủ, đó là FAIL.

Câu hỏi: {user_cmd}
Câu trả lời bot: {actual_resp}
Tài liệu tham khảo: {expected_ref[:600]}

Trả lời ĐÚNG định dạng, không thêm gì khác:
VERDICT: PASS hoặc VERDICT: FAIL"""


def run_judge(user_cmd: str, actual_resp: str, expected_ref: str) -> Dict[str, Any]:
    """Runs a single LLM judge call.

    Returns a dict with `resolved` (bool - False if the call failed, the
    circuit breaker is open, or the output could not be parsed; callers
    should fall back to the existing RETEST verdict in that case) and
    `verdict` ("PASS" | "FAIL") when resolved.
    """
    result: Dict[str, Any] = {
        "resolved": False,
        "verdict": None,
        "raw": "",
        "error": "",
    }

    if _circuit_is_open():
        result["error"] = "CIRCUIT_BREAKER_OPEN: skipping judge call - Ollama unresponsive."
        return result

    with _OLLAMA_CONCURRENCY:
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": JUDGE_MODEL,
                    "prompt": _build_judge_prompt(user_cmd, actual_resp, expected_ref),
                    "stream": False,
                    "options": {"num_predict": NUM_PREDICT},
                },
                timeout=REQUEST_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"Ollama error: {data['error']}")
            raw = (data.get("response") or "").strip()
            result["raw"] = raw
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            logger.warning(f"LLM judge call failed, falling back to RETEST: {result['error']}")
            _record_outcome(success=False)
            return result

    match = _VERDICT_RE.search(raw)
    if not match:
        result["error"] = "Judge output did not contain a parseable VERDICT line."
        _record_outcome(success=False)
        return result

    result["verdict"] = match.group(1).upper()
    result["resolved"] = True
    _record_outcome(success=True)
    print(f"🤖 LLM Judge: RETEST -> {result['verdict']}  (\"{user_cmd[:60]}\")")
    return result
