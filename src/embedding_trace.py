"""
Embedding Call Tracing Module.
Tracks every embedding-model invocation (OpenAI / HuggingFace) for performance
and cost monitoring: when it was called, how many texts, how long it took, and
whether it succeeded. This is separate from the evaluation trace_log in
eval_trace.py, which records test-case evaluation outcomes, not raw embedding
model calls.
"""

import time
import datetime
import logging
from collections import deque
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

MAX_LOG_ENTRIES = 2000
_EMBEDDING_CALL_LOG = deque(maxlen=MAX_LOG_ENTRIES)

_STATS: Dict[str, float] = {
    "total_calls": 0,
    "total_texts_embedded": 0,
    "total_errors": 0,
    "total_elapsed_ms": 0.0,
}


def _record(operation: str, provider: str, model_name: str, num_texts: int,
            elapsed_ms: float, success: bool, error: str = "") -> None:
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "operation": operation,      # "embed_documents" | "embed_query"
        "provider": provider,        # "openai" | "huggingface"
        "model_name": model_name,
        "num_texts": num_texts,
        "elapsed_ms": round(elapsed_ms, 1),
        "success": success,
        "error": error,
    }
    _EMBEDDING_CALL_LOG.append(entry)

    _STATS["total_calls"] += 1
    _STATS["total_texts_embedded"] += num_texts
    _STATS["total_elapsed_ms"] += elapsed_ms
    if not success:
        _STATS["total_errors"] += 1
        logger.error(
            f"Embedding call failed ({provider}/{model_name}, {operation}, {num_texts} texts): {error}"
        )


class TracedEmbeddings:
    """Wraps a LangChain-compatible embedding model, logging every call.

    Implements the same duck-typed interface LangChain embedding classes use
    (`embed_documents`, `embed_query`) and forwards everything else to the
    wrapped instance via __getattr__, so it can be used as a drop-in
    replacement anywhere an OpenAIEmbeddings/HuggingFaceEmbeddings instance
    is expected (e.g. passed to Chroma).
    """

    def __init__(self, inner: Any, provider: str, model_name: str):
        self._inner = inner
        self._provider = provider
        self._model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        start = time.time()
        try:
            result = self._inner.embed_documents(texts)
            _record("embed_documents", self._provider, self._model_name,
                     len(texts), (time.time() - start) * 1000, True)
            return result
        except Exception as e:
            _record("embed_documents", self._provider, self._model_name,
                     len(texts), (time.time() - start) * 1000, False,
                     f"{type(e).__name__}: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        start = time.time()
        try:
            result = self._inner.embed_query(text)
            _record("embed_query", self._provider, self._model_name,
                     1, (time.time() - start) * 1000, True)
            return result
        except Exception as e:
            _record("embed_query", self._provider, self._model_name,
                     1, (time.time() - start) * 1000, False,
                     f"{type(e).__name__}: {e}")
            raise

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_recent_calls(limit: int = 100) -> List[Dict[str, Any]]:
    """Returns the most recent embedding call log entries, newest first."""
    return list(_EMBEDDING_CALL_LOG)[-limit:][::-1]


def get_embedding_trace_summary() -> Dict[str, Any]:
    """Returns aggregate embedding call statistics for monitoring/dashboard use."""
    total_calls = int(_STATS["total_calls"])
    avg_latency = (_STATS["total_elapsed_ms"] / total_calls) if total_calls else 0.0
    return {
        "total_calls": total_calls,
        "total_texts_embedded": int(_STATS["total_texts_embedded"]),
        "total_errors": int(_STATS["total_errors"]),
        "error_rate_pct": round((_STATS["total_errors"] / total_calls) * 100, 2) if total_calls else 0.0,
        "avg_latency_ms": round(avg_latency, 1),
        "total_elapsed_sec": round(_STATS["total_elapsed_ms"] / 1000.0, 2),
    }


def reset_trace() -> None:
    """Clears the in-memory embedding call log and stats."""
    _EMBEDDING_CALL_LOG.clear()
    _STATS.update({
        "total_calls": 0,
        "total_texts_embedded": 0,
        "total_errors": 0,
        "total_elapsed_ms": 0.0,
    })
