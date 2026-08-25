"""
Evaluation Traceability & Severity Classification Module.
Provides trace logging, semantic error percentage calculation, 
and 3-tier severity classification (HIGH, MEDIUM, LOW).
"""

import uuid
import datetime
from typing import Dict, Any, List


class SeverityLevel:
    HIGH = "HIGH"         # Critical safety, false refusal, severe hallucination (Error 60.1% - 100%)
    MEDIUM = "MEDIUM"     # Functional inaccuracy, missing detail, wrong step (Error 30.1% - 60.0%)
    LOW = "LOW"           # Cosmetic, minor phrasing variation, extra filler (Error 15.1% - 30.0%)
    PASS = "PASS"         # Grounded, accurate, complete (Error 0.0% - 15.0%)


class ErrorCategory:
    FACT_HALLUCINATION = "FACT_HALLUCINATION"
    FALSE_REFUSAL = "FALSE_REFUSAL"
    STT_ACOUSTIC_MISMATCH = "STT_ACOUSTIC_MISMATCH"
    COMPLETENESS_LOSS = "COMPLETENESS_LOSS"
    NONE = "NONE"


def calculate_semantic_error_pct(sim_score: float) -> float:
    """Calculates Semantic Error Percentage from similarity score (0.0 to 1.0)."""
    bounded_sim = max(0.0, min(1.0, float(sim_score)))
    error_pct = (1.0 - bounded_sim) * 100.0
    return round(error_pct, 1)


def classify_severity(auto_result: str, sim_score: float, is_stt_mismatch: bool = False, is_false_refusal: bool = False) -> str:
    """Classifies evaluation failure into 3 Severity Tiers: HIGH, MEDIUM, LOW, or PASS."""
    if auto_result == "PASS" and sim_score >= 0.85:
        return SeverityLevel.PASS

    if is_stt_mismatch:
        return SeverityLevel.LOW  # Audio acoustic error, not bot logic bug

    if is_false_refusal or sim_score < 0.40:
        return SeverityLevel.HIGH

    if sim_score < 0.70:
        return SeverityLevel.MEDIUM

    return SeverityLevel.LOW


def generate_trace_log(
    test_id: str,
    user_cmd: str,
    vivi_listen: str,
    actual_resp: str,
    expected_resp: str,
    auto_result: str,
    sim_score: float,
    rule_info: str,
    rca: str,
    is_stt_mismatch: bool = False,
    is_false_refusal: bool = False,
    retrieved_chunks: List[Dict[str, Any]] = None,
    resolving_source: str = "",
    resolving_url: str = "",
    resolving_snippet: str = "",
    resolved_by: str = "",
    web_error: str = "",
    rag_error: str = ""
) -> Dict[str, Any]:
    """Generates a structured execution trace log object for 100% auditability & regression traceability."""
    import src.config as config

    trace_id = f"tr-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    error_pct = calculate_semantic_error_pct(sim_score)
    severity = classify_severity(auto_result, sim_score, is_stt_mismatch, is_false_refusal)

    error_cat = ErrorCategory.NONE
    if is_stt_mismatch:
        error_cat = ErrorCategory.STT_ACOUSTIC_MISMATCH
    elif is_false_refusal:
        error_cat = ErrorCategory.FALSE_REFUSAL
    elif auto_result == "FAIL":
        if sim_score < 0.3:
            error_cat = ErrorCategory.FACT_HALLUCINATION
        else:
            error_cat = ErrorCategory.COMPLETENESS_LOSS

    # Extract primary resolving chunk source if not explicitly provided
    if not resolving_source and retrieved_chunks:
        resolving_source = retrieved_chunks[0].get("source", "N/A")
        resolving_snippet = retrieved_chunks[0].get("snippet") or retrieved_chunks[0].get("content", "")[:200]
        if "url" in retrieved_chunks[0]:
            resolving_url = retrieved_chunks[0]["url"]

    return {
        "trace_id": trace_id,
        "test_id": test_id,
        "timestamp": datetime.datetime.now().isoformat(),
        # System & Model Versioning Traceability
        "system_version": "v2.1-agentic-eval",
        "embedding_model": getattr(config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "prompt_template_version": "v2.0-adaptive-5tier",
        "vector_db_version": "unified-chroma-22835",

        # Test Case Payload
        "user_cmd": user_cmd,
        "vivi_listen": vivi_listen,
        "stt_match": not is_stt_mismatch,
        "actual_resp": actual_resp,
        "expected_resp": expected_resp,
        "auto_result": auto_result,
        "sim_score": round(sim_score * 100.0, 1),
        "semantic_error_pct": error_pct,
        "severity": severity,
        "error_category": error_cat,

        # Auditability: Which Chunk or URL resolved the verdict
        "resolved_by": resolved_by or ("Explicit Spec" if expected_resp else ("RAG Vector DB" if retrieved_chunks else "Web Fact Verification")),
        "resolving_source": resolving_source,
        "resolving_url": resolving_url,
        "resolving_snippet": resolving_snippet,

        # Error & Exception Handling Audit
        "web_error": web_error,
        "rag_error": rag_error,

        "rule_info": rule_info,
        "rca": rca,
        "retrieved_chunks": retrieved_chunks or []
    }
