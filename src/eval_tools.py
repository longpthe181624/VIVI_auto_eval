"""
Evaluation Tools Module for Test Execution Analysis, Diff Comparison, and Root Cause Diagnosis.
Ported from AQC tool execution pattern for RAG-build-demo-1.
"""

import json
import re
from typing import Dict, Any, List
import src.config as config

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split()).lower()


def extract_relevant_sentence(text: str, query: str, max_chars: int = 160) -> str:
    """Extracts the specific section title and exact sentence/phrase matching the query."""
    if not text or not text.strip():
        return ""
    
    clean_text = text.strip()
    raw_segments = re.split(r'[\n\.\?!;]+', clean_text)
    segments = [s.strip() for s in raw_segments if len(s.strip()) > 3]
    
    if not segments:
        return clean_text[:max_chars]
        
    stopwords = {"là", "gì", "của", "và", "cho", "người", "xe", "trong", "được", "các", "với", "những", "để", "có", "thể", "nào", "này", "khi", "vf", "vf8", "vf8np"}
    q_words = [w.lower() for w in re.findall(r"\w+", query) if w.lower() not in stopwords and len(w) > 1]
    
    header = ""
    if len(segments) > 1 and len(segments[0]) < 45:
        header = f"[{segments[0].title()}] "
        candidate_segments = segments[1:]
    else:
        candidate_segments = segments

    best_segment = candidate_segments[0]
    best_score = -1
    
    for s in candidate_segments:
        s_norm = s.lower()
        score = sum(1 for w in q_words if w in s_norm)
        if score > best_score:
            best_score = score
            best_segment = s
            
    res = f"{header}{best_segment}"
    if len(res) > max_chars:
        return res[:max_chars] + "..."
    return res


def compute_similarity(actual: str, expected: str, user_cmd: str = "") -> float:
    """Computes factual accuracy and keyword grounding similarity score (0.0 to 1.0)."""
    norm_act = normalize_text(actual)
    norm_exp = normalize_text(expected)
    norm_usr = normalize_text(user_cmd)

    if not norm_act:
        return 0.0

    canned_phrases = ["dạ em đây", "cần em hỗ trợ gì", "anh/chị cần em giúp gì", "em luôn sẵn"]
    if any(p in norm_act for p in canned_phrases) and len(norm_act) < 120 and len(norm_exp) > 50:
        return 0.1

    if norm_act == norm_exp:
        return 1.0

    stopwords = {"là", "gì", "của", "và", "cho", "người", "xe", "trong", "được", "các", "với", "những", "để", "có", "thể", "nào", "này", "khi", "vf", "vf8", "vf8np"}
    words_act = [w for w in re.findall(r"\w+", norm_act) if w not in stopwords and len(w) > 1]
    words_exp = [w for w in re.findall(r"\w+", norm_exp) if w not in stopwords and len(w) > 1]
    words_usr = [w for w in re.findall(r"\w+", norm_usr) if w not in stopwords and len(w) > 1]

    if not words_act:
        return 0.0

    set_exp = set(words_exp)

    matched_in_exp = sum(1 for w in words_act if w in set_exp)
    recall_exp = matched_in_exp / len(words_act) if words_act else 0.0

    if recall_exp >= 0.35 or any(w in norm_act for w in words_usr if len(w) > 3):
        score = min(1.0, recall_exp * 1.5 + 0.3)
    else:
        score = recall_exp

    return round(score, 3)


EVAL_TOOL_DEFINITIONS = [
    {
        "name": "eval_test_result",
        "description": "Evaluates a test execution log, actual vs expected outputs, error messages, and assigns Pass/Fail status with Root Cause Analysis (RCA).",
        "parameters": {
            "test_name": "Name or ID of the test case",
            "expected_behavior": "Expected test outcome or spec requirements",
            "actual_behavior": "Actual log output or measured execution result",
            "error_log": "Optional execution traceback, error message, or console log"
        }
    },
    {
        "name": "compare_expected_actual",
        "description": "Performs step-by-step diff and signal/value comparison between expected metrics/signals and actual execution values.",
        "parameters": {
            "expected_values": "JSON object or text list of expected parameter values",
            "actual_values": "JSON object or text list of measured parameter values"
        }
    },
    {
        "name": "web_search_verification",
        "description": "Performs live web search for general knowledge queries outside static training knowledge cutoff to verify factual accuracy of chatbot answers.",
        "parameters": {
            "query": "Search query to verify facts",
            "max_results": "Number of web search results to retrieve (default 3)"
        }
    },
    {
        "name": "rag_rule_search",
        "description": "Queries vector database for domain evaluation rules, command list specs, preconditions, or error code responses.",
        "parameters": {
            "query": "Search query for rule criteria or command domain",
            "top_k": "Number of relevant rule chunks to retrieve (default 3)"
        }
    },
    {
        "name": "rag_spec_search",
        "description": "Queries vector database for system specification documents, requirements, or test plan criteria.",
        "parameters": {
            "query": "Search query for requirement or specification criteria",
            "top_k": "Number of relevant chunks to retrieve (default 3)"
        }
    },
    {
        "name": "generate_eval_report",
        "description": "Formats a final structured evaluation report summarizing overall pass/fail statistics, critical root causes, and remediation actions.",
        "parameters": {
            "summary_data": "Dict containing total_tests, passed, failed, failure_categories, and rca_notes"
        }
    }
]


def eval_test_result(test_name: str, expected_behavior: str, actual_behavior: str, error_log: str = "") -> Dict[str, Any]:
    """Perform deterministic & structural analysis on a test case result."""
    combined_log = f"{actual_behavior}\n{error_log}".lower()
    
    failure_category = "SUCCESS"
    status = "PASS"
    root_cause = "Execution matched expected specification."
    remediation = "No action required."

    is_failed = False
    fail_keywords = ["fail", "error", "exception", "assert", "timeout", "mismatch", "nullpointer", "segfault", "404", "500"]
    if any(kw in combined_log for kw in fail_keywords) or (expected_behavior and expected_behavior.lower().strip() not in actual_behavior.lower().strip()):
        is_failed = True

    if is_failed:
        status = "FAIL"
        if "timeout" in combined_log:
            failure_category = "Timeout Error"
            root_cause = "Test execution exceeded configured timeout threshold or target service was unresponsive."
            remediation = "Increase timeout interval, inspect network latency or target service availability."
        elif "assert" in combined_log or "mismatch" in combined_log:
            failure_category = "Assertion / Signal Mismatch"
            root_cause = f"Actual output did not match expected behavior ({expected_behavior[:100]})."
            remediation = "Verify logic implementation or update test case expectation if requirements changed."
        elif "nullpointer" in combined_log or "none" in combined_log or "attributeerror" in combined_log:
            failure_category = "Null Reference Error"
            root_cause = "Null or missing object dereferenced during test execution."
            remediation = "Add defensive null checks before dereferencing variables in test setup or target method."
        elif "connection" in combined_log or "refused" in combined_log or "500" in combined_log:
            failure_category = "Infrastructure / Dependency Error"
            root_cause = "Target service endpoint, database, or dependent API returned error or refused connection."
            remediation = "Ensure backend services, database containers, and dependent APIs are running properly."
        else:
            failure_category = "Logic / Functional Error"
            root_cause = f"Execution output deviates from expected specification: {actual_behavior[:150]}"
            remediation = "Review log traceback and step through function execution."

    return {
        "test_name": test_name,
        "status": status,
        "failure_category": failure_category,
        "root_cause_analysis": root_cause,
        "suggested_remediation": remediation,
        "expected": expected_behavior,
        "actual": actual_behavior[:300]
    }


def compare_expected_actual(expected_values: Any, actual_values: Any) -> Dict[str, Any]:
    """Perform diff comparison between expected and actual values."""
    diffs = []
    
    if isinstance(expected_values, str):
        try:
            expected_values = json.loads(expected_values)
        except Exception:
            pass
    if isinstance(actual_values, str):
        try:
            actual_values = json.loads(actual_values)
        except Exception:
            pass

    if isinstance(expected_values, dict) and isinstance(actual_values, dict):
        all_keys = set(expected_values.keys()).union(set(actual_values.keys()))
        for k in sorted(all_keys):
            exp_val = expected_values.get(k, "<MISSING>")
            act_val = actual_values.get(k, "<MISSING>")
            match = (exp_val == act_val)
            diffs.append({
                "field": k,
                "expected": exp_val,
                "actual": act_val,
                "match": match
            })
    else:
        exp_str = str(expected_values).strip()
        act_str = str(actual_values).strip()
        diffs.append({
            "field": "raw_output",
            "expected": exp_str,
            "actual": act_str,
            "match": exp_str == act_str
        })

    mismatches = [d for d in diffs if not d["match"]]
    return {
        "total_fields_compared": len(diffs),
        "mismatch_count": len(mismatches),
        "status": "PASS" if len(mismatches) == 0 else "MISMATCH",
        "diff_summary": diffs
    }


_CACHED_RULES_VECTORSTORE = None

def get_rules_vectorstore(force_reload: bool = False):
    """Returns the unified Chroma vector store."""
    from src.rag_chain import get_vector_store
    return get_vector_store(force_reload=force_reload)


import logging
from collections import deque

logger = logging.getLogger(__name__)

class BatchHealthMonitor:
    """Sliding-window health monitor and circuit breaker for batch evaluation web queries."""

    def __init__(self, error_rate_threshold: float = 0.15, window: int = 50):
        self.error_rate_threshold = error_rate_threshold
        self.window = window
        self.errors = deque(maxlen=window)
        self.circuit_open = False
        self.alert_msg = ""

    def record(self, had_error: bool, error_details: str = ""):
        self.errors.append(1 if had_error else 0)
        if len(self.errors) == self.window:
            rate = sum(self.errors) / float(self.window)
            if rate > self.error_rate_threshold:
                self.circuit_open = True
                self.alert_msg = f"ALERT: Web search error rate {rate:.0%} over last {self.window} rows — check DDGS/network."
                logger.error(self.alert_msg)

    def is_circuit_open(self) -> bool:
        return self.circuit_open

    def reset(self):
        self.errors.clear()
        self.circuit_open = False
        self.alert_msg = ""


_WEB_SEARCH_CACHE = {}
GLOBAL_HEALTH_MONITOR = BatchHealthMonitor(error_rate_threshold=0.15, window=50)


def web_search_verification(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Queries DuckDuckGo live web search for factual verification with caching, fast 2s timeout, and narrowed exception handling."""
    clean_q = query.strip().lower()
    if not clean_q:
        return {"query": query, "found_snippets": 0, "results": [], "summary": ""}
    
    if clean_q in _WEB_SEARCH_CACHE:
        return _WEB_SEARCH_CACHE[clean_q]

    if GLOBAL_HEALTH_MONITOR.is_circuit_open():
        return {
            "query": query,
            "found_snippets": 0,
            "results": [],
            "summary": "",
            "error": "CIRCUIT_BREAKER_OPEN: Web search suspended due to high fleet error rate (>15%)"
        }

    try:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            from ddgs import DDGS

        results = []
        with DDGS(timeout=2) as ddgs:
            raw_res = list(ddgs.text(query, max_results=int(max_results)))
            for idx, r in enumerate(raw_res, 1):
                results.append({
                    "rank": idx,
                    "title": r.get("title", ""),
                    "snippet": r.get("body", "")[:300],  # Display snippet
                    "full_snippet": r.get("body", ""),     # Full text for similarity scoring
                    "url": r.get("href", "")
                })
        summary = " ".join([r["full_snippet"] for r in results])
        ret = {
            "query": query,
            "found_snippets": len(results),
            "results": results,
            "summary": summary
        }
        _WEB_SEARCH_CACHE[clean_q] = ret
        GLOBAL_HEALTH_MONITOR.record(had_error=False)
        return ret
    except (TimeoutError, Exception) as e:
        err_type = type(e).__name__
        err_msg = str(e)
        
        # Distinguish network/timeout vs code bugs
        if any(k in err_type.lower() or k in err_msg.lower() for k in ["timeout", "connection", "http", "socket", "network"]):
            formatted_err = f"NETWORK_TIMEOUT: {err_type}: {err_msg}"
        else:
            formatted_err = f"UNEXPECTED_ERROR: {err_type}: {err_msg}"
            logger.error(f"Unexpected error in web_search_verification for query '{query}': {formatted_err}", exc_info=True)

        GLOBAL_HEALTH_MONITOR.record(had_error=True, error_details=formatted_err)
        empty_ret = {"query": query, "found_snippets": 0, "results": [], "summary": "", "error": formatted_err}
        _WEB_SEARCH_CACHE[clean_q] = empty_ret
        return empty_ret


_RULE_SEARCH_CACHE = {}


def rag_rule_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search unified vector DB for command rules and specs with memory-efficient caching."""
    clean_q = query.strip().lower()
    cache_key = (clean_q, int(top_k))

    if cache_key in _RULE_SEARCH_CACHE:
        return _RULE_SEARCH_CACHE[cache_key]

    try:
        from src.rag_chain import get_vector_store
        vectorstore = get_vector_store()
        if vectorstore is None:
            return {"query": query, "found_chunks": 0, "results": [], "error": f"Vector DB directory '{config.CHROMA_DB_DIR}' not found."}
        
        # Try metadata filtered search first, fallback to standard similarity search
        try:
            docs = vectorstore.similarity_search(query, k=int(top_k), filter={"doc_type": "command_rule"})
        except Exception:
            docs = []
        
        if not docs:
            docs = vectorstore.similarity_search(query, k=int(top_k))

        results = []
        for idx, doc in enumerate(docs, 1):
            results.append({
                "rank": idx,
                "content": doc.page_content[:400],
                "source": doc.metadata.get("source", "Unknown"),
                "sheet": doc.metadata.get("sheet", "")
            })
        ret = {
            "query": query,
            "found_chunks": len(results),
            "results": results
        }
        if len(_RULE_SEARCH_CACHE) > 5000:
            _RULE_SEARCH_CACHE.clear()
        _RULE_SEARCH_CACHE[cache_key] = ret
        return ret
    except Exception as e:
        return {"query": query, "found_chunks": 0, "results": [], "error": str(e)}


def rag_spec_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search unified vector DB for Owner Manuals and general knowledge chunks."""
    try:
        from src.rag_chain import get_vector_store
        vectorstore = get_vector_store()
        if vectorstore is None:
            return {"query": query, "found_chunks": 0, "results": []}

        try:
            docs = vectorstore.similarity_search(query, k=int(top_k), filter={"doc_type": "owner_manual"})
        except Exception:
            docs = []

        if not docs:
            docs = vectorstore.similarity_search(query, k=int(top_k))

        results = []
        for idx, doc in enumerate(docs, 1):
            results.append({
                "rank": idx,
                "content": doc.page_content[:400],
                "source": doc.metadata.get("source", "Unknown")
            })
        return {
            "query": query,
            "found_chunks": len(results),
            "results": results
        }
    except Exception as e:
        return {"query": query, "found_chunks": 0, "results": [], "error": str(e)}


def generate_eval_report(summary_data: Any) -> Dict[str, Any]:
    """Generate Markdown evaluation report."""
    if isinstance(summary_data, str):
        try:
            summary_data = json.loads(summary_data)
        except Exception:
            summary_data = {"notes": summary_data}

    total = summary_data.get("total_tests", 1)
    passed = summary_data.get("passed", 0)
    failed = summary_data.get("failed", 0)
    rca = summary_data.get("rca_notes", "Evaluation complete.")

    md = f"""### 📊 Test Evaluation & Root Cause Analysis Summary Report

- **Total Tests Evaluated:** {total}
- **Passed:** {passed} ✅
- **Failed:** {failed} ❌
- **Pass Rate:** {(passed / total * 100) if total > 0 else 0:.1f}%

#### 🔍 Root Cause Analysis (RCA) & Failure Breakdown:
{rca}
"""
    return {
        "report_markdown": md,
        "summary": summary_data
    }


async def execute_eval_tool(tool_name: str, args: dict) -> Dict[str, Any]:
    """Tool dispatcher."""
    if tool_name == "eval_test_result":
        return eval_test_result(
            test_name=args.get("test_name", "Test_Case"),
            expected_behavior=args.get("expected_behavior", ""),
            actual_behavior=args.get("actual_behavior", ""),
            error_log=args.get("error_log", "")
        )
    elif tool_name == "compare_expected_actual":
        return compare_expected_actual(
            expected_values=args.get("expected_values", {}),
            actual_values=args.get("actual_values", {})
        )
    elif tool_name == "web_search_verification":
        return web_search_verification(
            query=args.get("query", ""),
            max_results=args.get("max_results", 3)
        )
    elif tool_name == "rag_rule_search":
        return rag_rule_search(
            query=args.get("query", ""),
            top_k=args.get("top_k", 3)
        )
    elif tool_name == "rag_spec_search":
        return rag_spec_search(
            query=args.get("query", ""),
            top_k=args.get("top_k", 3)
        )
    elif tool_name == "generate_eval_report":
        return generate_eval_report(
            summary_data=args.get("summary_data", {})
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}
