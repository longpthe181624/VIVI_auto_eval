"""
Test Evaluation AI Agent Module.
Ported from AQC agentic-rag/agent.py multi-turn loop, reflection & clarification handlers for RAG-build-demo-1.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, Any, List, Tuple
import src.config as config
from src.eval_tools import EVAL_TOOL_DEFINITIONS, execute_eval_tool

logger = logging.getLogger(__name__)

CLARIFY_PREFIX = "CLARIFY:"
REFLECT_PREFIX = "REFLECT:"
MAX_LOOP = 7
MAX_HISTORY_TURNS = 10
MAX_TOOL_RESULT_CHARS = 3000

_TOOL_CALL_LINE_RE = re.compile(r"^\s*(?:TOOL_CALL|TOOL_CODE)\s*:.*$", re.IGNORECASE)


def _strip_residual_tool_calls(text: str) -> str:
    if not text:
        return text
    lines = [ln for ln in text.split("\n") if not _TOOL_CALL_LINE_RE.match(ln)]
    return "\n".join(lines).strip()


def _build_tools_desc() -> str:
    return json.dumps(EVAL_TOOL_DEFINITIONS, indent=2, ensure_ascii=False)


def _tool_names() -> str:
    return ", ".join(t["name"] for t in EVAL_TOOL_DEFINITIONS)


def _system_prompt() -> str:
    tools_desc = _build_tools_desc()
    names = _tool_names()
    return f"""You are the Test Evaluation & Root Cause Analysis (RCA) AI Agent.

### Available Tools ({names}):
{tools_desc}

### EXECUTION PROTOCOL:
Call tools when analyzing test execution results, logs, expected vs actual outputs, or requirement specs.
Format: TOOL_CALL: <name> | {{"key": "value"}}
EXAMPLE: TOOL_CALL: eval_test_result | {{"test_name": "TC_01", "expected_behavior": "Status 200 OK", "actual_behavior": "Status 500 Internal Server Error", "error_log": "NullPointerInController"}}
EXAMPLE 2: TOOL_CALL: compare_expected_actual | {{"expected_values": {{"voltage": 12.0}}, "actual_values": {{"voltage": 10.5}}}}

### RULES:
1. ALWAYS call `eval_test_result` or `compare_expected_actual` when provided with test logs or parameter pairs.
2. Structure your final output using Markdown headers, Pass/Fail status badges, Root Cause Analysis (RCA), and Remediation steps.
3. If logs/specs are ambiguous or incomplete → use `{CLARIFY_PREFIX} <question>` to request specific missing log details or requirements.
4. If the user reports an evaluation was incorrect ("sai", "nhầm", "wrong") → analyze why the previous evaluation failed, call tool with adjusted parameters, and fix output.
"""


def _get_llm():
    """Gets initialized LLM or local Ollama model."""
    if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=0.2,
            openai_api_key=config.OPENAI_API_KEY
        )
    else:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model="qwen2.5:3b", temperature=0.2)
        except Exception:
            return None


async def _generate_llm_response(prompt_text: str) -> str:
    llm = _get_llm()
    if llm:
        try:
            res = await llm.ainvoke(prompt_text)
            return res.content
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}. Using deterministic tool runner fallback.")

    # Offline / Fallback generator for test evaluation
    # Auto-detect test evaluation queries and emit appropriate tool call if LLM API is unavailable
    if "TOOL_CALL:" not in prompt_text and ("expected" in prompt_text.lower() or "actual" in prompt_text.lower() or "log" in prompt_text.lower()):
        return f'TOOL_CALL: eval_test_result | {{"test_name": "Test_Execution", "expected_behavior": "Expected result matched spec", "actual_behavior": "{prompt_text[:200].replace(chr(10), " ")}", "error_log": ""}}'
    return f"Test Evaluation Engine Ready. Received query: {prompt_text[:100]}..."


def _extract_tool_calls(text: str) -> List[Tuple[str, dict, str]]:
    calls = []
    seen = set()

    for line in text.strip().split("\n"):
        line_str = line.strip()
        m = re.match(r"^(?:TOOL_CALL|TOOL_CODE):\s*(\w+)\s*(?:\|\s*(\{.*\}|.*))?$", line_str, re.DOTALL)
        if m:
            name = m.group(1)
            args_raw = m.group(2)
            args = {}
            if args_raw and args_raw.startswith("{"):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            key = (name, json.dumps(args, sort_keys=True))
            if key not in seen:
                seen.add(key)
                calls.append((name, args, line_str))

    return calls


def _detect_reflection(user_message: str) -> Dict[str, Any] | None:
    lower = user_message.lower().strip()
    triggers = ["sai", "không đúng", "nhầm", "wrong", "incorrect", "error in eval", "sai rồi"]
    for t in triggers:
        if t in lower:
            return {
                "hint": f"User reported previous evaluation was incorrect: '{user_message}'. Re-evaluate logs and specs.",
            }
    return None


class TestEvalAgent:
    """Multi-turn Test Evaluation Agent engine with reflection and clarification loops."""

    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def clear_history(self):
        self.history = []

    async def process_message(self, user_message: str) -> Dict[str, Any]:
        reflect_info = _detect_reflection(user_message)
        if reflect_info:
            prompt_context = (
                f"[SYSTEM REFLECTION]: User reported '{user_message}'. "
                f"Analyze previous evaluation error and re-run tools with adjusted logic."
            )
            self.history.append({"role": "user", "content": prompt_context})
        else:
            self.history.append({"role": "user", "content": user_message})

        history_summary = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in self.history[-MAX_HISTORY_TURNS:])
        prompt = f"{_system_prompt()}\n\nCONVERSATION HISTORY:\n{history_summary}\n\nUser Question:\n{user_message}\n\nAssistant:"

        ai_response = await _generate_llm_response(prompt)
        self.history.append({"role": "assistant", "content": ai_response})

        all_tool_calls = []
        for loop_i in range(MAX_LOOP):
            tool_calls = _extract_tool_calls(ai_response)

            if ai_response.strip().startswith(CLARIFY_PREFIX):
                break

            if ai_response.strip().startswith(REFLECT_PREFIX):
                break

            if not tool_calls:
                break

            for name, args, raw in tool_calls:
                result_dict = await execute_eval_tool(name, args)
                entry = {"name": name, "args": args, "result": result_dict}
                all_tool_calls.append(entry)

            result_blocks = [
                f"Tool Result ({tc['name']}): {json.dumps(tc['result'], ensure_ascii=False, indent=2)[:MAX_TOOL_RESULT_CHARS]}"
                for tc in all_tool_calls[-len(tool_calls):]
            ]
            result_str = "\n\n".join(result_blocks)

            next_prompt = (
                f"{_system_prompt()}\n\n"
                f"TOOL EXECUTION RESULTS:\n{result_str}\n\n"
                f"Synthesize the evaluation results into a clear final report with Pass/Fail status, RCA, and remediation steps:"
            )
            ai_response = await _generate_llm_response(next_prompt)
            self.history.append({"role": "assistant", "content": ai_response})

        ai_response = _strip_residual_tool_calls(ai_response)
        
        # Fallback formatting if tool calls were executed but AI output is minimal
        if not ai_response or len(ai_response.strip()) < 30:
            if all_tool_calls:
                last_res = all_tool_calls[-1]["result"]
                if "status" in last_res:
                    ai_response = (
                        f"### 🧪 Test Evaluation Result\n"
                        f"- **Test Name:** {last_res.get('test_name', 'Test Case')}\n"
                        f"- **Status:** {last_res.get('status', 'N/A')}\n"
                        f"- **Failure Category:** {last_res.get('failure_category', 'None')}\n"
                        f"- **Root Cause:** {last_res.get('root_cause_analysis', 'N/A')}\n"
                        f"- **Remediation:** {last_res.get('suggested_remediation', 'N/A')}\n"
                    )

        return {
            "reply": ai_response,
            "tool_calls": all_tool_calls
        }
