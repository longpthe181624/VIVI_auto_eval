"""
Autonomous Agent Execution Loop Module.
Handles multi-step self-correction, query expansion, and verification loops for ambiguous test cases.
"""

from typing import Dict, Any, List
from src.eval_tools import rag_rule_search, rag_spec_search, web_search_verification, compute_similarity


import src.config as config

class AgentEvalLoop:
    """Orchestrates autonomous multi-step evaluation loops for ambiguous or borderline test cases."""

    def __init__(self, max_iterations: int = 2):
        self.max_iterations = max_iterations

    def run_correction_loop(self, user_cmd: str, actual_resp: str, initial_score: float) -> Dict[str, Any]:
        """Runs iterative self-correction loop when initial similarity score is borderline (0.25 to 0.45)."""
        current_score = initial_score
        iterations_run = 0
        loop_logs = []
        rag_err = ""
        web_err = ""

        # Iteration 1: Query Expansion & Keyword Isolation RAG search
        iterations_run += 1
        expanded_query = f"{user_cmd} thông số tính năng vận hành hướng dẫn".strip()
        try:
            rag_res = rag_rule_search(query=expanded_query, top_k=2)
            chunks = rag_res.get("results", [])
            if rag_res.get("error"):
                rag_err = str(rag_res["error"])
        except Exception as e:
            chunks = []
            rag_err = f"UNEXPECTED_RAG_ERROR: {type(e).__name__}: {e}"

        if chunks:
            top_content = chunks[0].get("content", "")  # FULL UNTRUNCATED TEXT for similarity scoring
            src_name = chunks[0].get("source", "RAG Vector DB")
            score_iter1 = compute_similarity(actual_resp, top_content, user_cmd)  # Score on FULL text
            loop_logs.append(f"Iteration 1 (RAG Expansion: {src_name}): Score {score_iter1 * 100:.1f}%")
            
            if score_iter1 >= config.RAG_PASS_THRESHOLD:
                display_snippet = top_content[:200]  # Truncate ONLY for storage/readability
                return {
                    "resolved": True,
                    "final_score": round(score_iter1 * 100.0, 1),
                    "auto_result": "PASS",
                    "iterations": iterations_run,
                    "loop_logs": loop_logs,
                    "matched_content": display_snippet,
                    "resolved_by": "Agent Loop Iteration 1 (RAG Search)",
                    "resolving_source": src_name,
                    "resolving_url": "",
                    "resolving_snippet": display_snippet,
                    "rag_error": rag_err,
                    "web_error": web_err
                }

        # Iteration 2: Live Web Search Verification Fallback
        if iterations_run < self.max_iterations:
            iterations_run += 1
            try:
                web_res = web_search_verification(query=user_cmd, max_results=2)
                results = web_res.get("results", [])
                if web_res.get("error"):
                    web_err = str(web_res["error"])
            except Exception as e:
                results = []
                web_err = f"UNEXPECTED_WEB_ERROR: {type(e).__name__}: {e}"

            if results:
                # Score on FULL UNTRUNCATED snippets combined
                combined_web = " ".join([r.get("full_snippet") or r.get("snippet", "") for r in results])
                score_iter2 = compute_similarity(actual_resp, combined_web, user_cmd)  # Score on FULL text
                loop_logs.append(f"Iteration 2 (Web Search Fallback): Score {score_iter2 * 100:.1f}%")
                
                # Option A: Uncurated Web Fallback requires stricter bar (0.50)
                if score_iter2 >= config.WEB_PASS_THRESHOLD:
                    top_url = results[0].get("url", "")
                    top_title = results[0].get("title", "Web Search Result")
                    top_snippet = (results[0].get("full_snippet") or results[0].get("snippet", ""))[:200]  # Truncate ONLY for logging
                    return {
                        "resolved": True,
                        "final_score": round(max(75.0, score_iter2 * 100.0), 1),
                        "auto_result": "PASS",
                        "iterations": iterations_run,
                        "loop_logs": loop_logs,
                        "matched_content": top_snippet,
                        "resolved_by": "Agent Loop Iteration 2 (Live Web Search)",
                        "resolving_source": f"Web Search ({top_title})",
                        "resolving_url": top_url,
                        "resolving_snippet": top_snippet,
                        "rag_error": rag_err,
                        "web_error": web_err
                    }

        return {
            "resolved": False,
            "final_score": round(current_score * 100.0, 1),
            "auto_result": "RETEST",
            "iterations": iterations_run,
            "loop_logs": loop_logs,
            "matched_content": "",
            "resolved_by": "Unresolved (High Uncertainty)",
            "resolving_source": "",
            "resolving_url": "",
            "resolving_snippet": "",
            "rag_error": rag_err,
            "web_error": web_err
        }
