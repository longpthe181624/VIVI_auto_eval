import os
import sys
import re
import json
import gc
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root and site-packages are in sys.path
project_root = Path(__file__).resolve().parent.parent
site_packages = project_root / "myenv" / "lib" / "python3.14" / "site-packages"
if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.insert(0, str(site_packages))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
import src.config as config
from src.eval_tools import rag_rule_search, eval_test_result, web_search_verification
from src.test_eval_agent import TestEvalAgent


# Fill styles for output Excel formatting
PASS_FILL = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")  # Light green
FAIL_FILL = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")  # Light red
RETEST_FILL = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")  # Light yellow
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")  # Navy blue
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")


def _detect_columns(header_row: List[Any]) -> Dict[str, int]:
    """Dynamically maps Excel column headers to standard evaluation fields."""
    col_map = {}
    if not header_row:
        return col_map

    header_strs = [str(c).strip().lower() if c is not None else "" for c in header_row]

    for idx, h in enumerate(header_strs):
        if not h:
            continue
        if any(k in h for k in ["auto_eval_result", "auto_eval"]):
            col_map.setdefault("auto_eval_start", idx)
        elif any(k in h for k in ["name", "testcase", "test_case", "id", "tc"]):
            col_map.setdefault("name", idx)
        elif any(k in h for k in ["user_command", "command", "prompt", "query", "question", "câu_lệnh"]):
            col_map.setdefault("user_command", idx)
        elif any(k in h for k in ["actual_resp", "actual", "response", "model_answer", "phản_hồi_thực_tế"]):
            col_map.setdefault("actual_resp", idx)
        elif any(k in h for k in ["expected_resp", "expected", "ground_truth", "kết_quả_mong_muốn", "target"]):
            col_map.setdefault("expected_resp", idx)
        elif any(k in h for k in ["vivi_listen", "listen", "transcribed", "input"]):
            col_map.setdefault("vivi_listen", idx)
        elif any(k in h for k in ["result", "status", "trạng_thái"]):
            col_map.setdefault("prev_result", idx)
        elif any(k in h for k in ["latency", "time", "duration"]):
            col_map.setdefault("latency", idx)

    return col_map


def trim_trailing_empty(row_values: tuple) -> list:
    """Trims trailing None or empty whitespace values from an Excel row tuple dynamically."""
    if not row_values:
        return []
    last_idx = -1
    for idx in range(len(row_values) - 1, -1, -1):
        v = row_values[idx]
        if v is not None and str(v).strip() != "":
            last_idx = idx
            break
    if last_idx == -1:
        return []
    return list(row_values[:last_idx + 1])


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Remove extra whitespace and lower case for baseline matching
    return " ".join(text.strip().split()).lower()


def compute_similarity(actual: str, expected: str, user_cmd: str = "") -> float:
    """Computes factual accuracy and keyword grounding similarity score (0.0 to 1.0)."""
    norm_act = normalize_text(actual)
    norm_exp = normalize_text(expected)
    norm_usr = normalize_text(user_cmd)

    if not norm_act:
        return 0.0

    # Generic canned fallback detector (e.g. "Dạ, em đây! Anh/chị cần em hỗ trợ gì không ạ?")
    canned_phrases = ["dạ em đây", "cần em hỗ trợ gì", "anh/chị cần em giúp gì", "em luôn sẵn"]
    if any(p in norm_act for p in canned_phrases) and len(norm_act) < 120 and len(norm_exp) > 50:
        return 0.1

    if norm_act == norm_exp:
        return 1.0

    # Token extraction (ignoring short stopwords)
    stopwords = {"là", "gì", "của", "và", "cho", "người", "xe", "trong", "được", "các", "với", "những", "để", "có", "thể", "nào", "này", "khi", "vf", "vf8", "vf8np"}
    words_act = [w for w in re.findall(r"\w+", norm_act) if w not in stopwords and len(w) > 1]
    words_exp = [w for w in re.findall(r"\w+", norm_exp) if w not in stopwords and len(w) > 1]
    words_usr = [w for w in re.findall(r"\w+", norm_usr) if w not in stopwords and len(w) > 1]

    if not words_act:
        return 0.0

    # Key answer overlap: check how many key tokens in Actual exist in Expected or User query context
    set_exp = set(words_exp)
    set_usr = set(words_usr)

    matched_in_exp = sum(1 for w in words_act if w in set_exp)
    recall_exp = matched_in_exp / len(words_act) if words_act else 0.0

    # Containment boost if key user query terms + expected terms match actual response
    if recall_exp >= 0.35 or any(w in norm_act for w in words_usr if len(w) > 3):
        score = min(1.0, recall_exp * 1.5 + 0.3)
    else:
        score = recall_exp

    return round(score, 3)


def should_trigger_web_search(user_cmd: str, actual_resp: str) -> bool:
    """Determines whether a test case query requires live web search factual verification."""
    norm_usr = normalize_text(user_cmd)
    norm_act = normalize_text(actual_resp)

    # 1. Bot response explicitly indicates web search synthesis
    web_trigger_response_terms = ["tìm kiếm", "trên mạng", "theo tin tức", "theo nguồn tin", "kết quả tìm kiếm", "trực tuyến", "trích dẫn"]
    if any(term in norm_act for term in web_trigger_response_terms):
        return True

    # 2. Query contains explicit real-world news, celebrity, or temporal fact indicators
    web_trigger_query_terms = [
        "ai là", "ra tù", "bị bắt", "scandal", "bạo mẫu", "gian lận", "bài hát", "phim", "năm 202", "năm 201", "tin tức",
        "thì sao", "ở đâu", "diễn viên", "thủ tướng", "chủ tịch", "tổng thống", "vô địch", "giải đấu", "khi nào"
    ]
    if any(term in norm_usr for term in web_trigger_query_terms):
        return True

    return False


from src.eval_router import AdaptiveEvalRouter


class ExcelTestEvaluator:
    """Automated RAG-enabled Excel test case evaluator."""

    def __init__(self):
        self.eval_agent = TestEvalAgent()
        self.router = AdaptiveEvalRouter()

    def evaluate_row_sync(
        self,
        name: str,
        user_cmd: str,
        actual_resp: str,
        expected_resp: str
    ) -> Dict[str, Any]:
        """Evaluates a single test case row via multi-tiered adaptive routing."""
        return self.router.evaluate(name, user_cmd, actual_resp, expected_resp)

    def evaluate_file(self, input_excel_path: str, output_excel_path: str = None) -> str:
        """Evaluates all test cases in an Excel file using lightweight streaming mode to prevent memory bloat."""
        input_path = Path(input_excel_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input Excel file '{input_excel_path}' not found.")

        print(f"\n🧪 Starting Automated Evaluation on: {input_path.name}")
        try:
            wb_in = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
        except Exception as e:
            raise ValueError(f"Could not open Excel file {input_excel_path}: {e}")

        wb_out = openpyxl.Workbook()
        # Remove default empty sheet
        wb_out.remove(wb_out.active)

        total_eval_count = 0
        pass_count = 0
        fail_count = 0
        retest_count = 0

        new_headers = ["Auto_Eval_Result", "Similarity_Score(%)", "Matched_Rule_Spec", "Root_Cause_Analysis", "Suggested_Remediation"]

        for sheet_name in wb_in.sheetnames:
            ws_in = wb_in[sheet_name]
            ws_out = wb_out.create_sheet(title=sheet_name)

            header_vals = None
            header_row_num = -1
            col_map = {}
            raw_rows_buffer = []

            # Stream rows to find header and limit column bounds dynamically
            for r_idx, row in enumerate(ws_in.iter_rows(values_only=True), 1):
                trimmed_row = trim_trailing_empty(row)
                if not trimmed_row:
                    continue

                if header_vals is None:
                    test_map = _detect_columns(trimmed_row)
                    if "user_command" in test_map or "actual_resp" in test_map:
                        header_vals = [str(c).strip() if c is not None else "" for c in trimmed_row]
                        col_map = test_map
                        header_row_num = r_idx
                        continue
                    else:
                        # Copy pre-header rows (e.g. metadata/title block) directly
                        ws_out.append(trimmed_row)
                else:
                    raw_rows_buffer.append(trimmed_row)

            if not header_vals or ("user_command" not in col_map and "actual_resp" not in col_map):
                # Sheet has no test case headers
                continue

            print(f"  📊 Processing Sheet: '{sheet_name}' ({len(raw_rows_buffer)} test cases)...")

            # Determine actual valid column count
            max_data_col = max(col_map.values()) if col_map else len(header_vals) - 1
            max_col_idx = min(len(header_vals), max_data_col + 5)
            clean_header = header_vals[:max_col_idx]

            # Output header setup
            if "auto_eval_start" in col_map:
                col_start_idx = col_map["auto_eval_start"] + 1
            else:
                col_start_idx = len(clean_header) + 1

            full_header = clean_header + new_headers
            ws_out.append(full_header)

            # Format header cells
            for c_idx in range(1, len(full_header) + 1):
                cell = ws_out.cell(row=ws_out.max_row, column=c_idx)
                if c_idx >= col_start_idx:
                    cell.fill = HEADER_FILL
                    cell.font = HEADER_FONT
                    cell.alignment = Alignment(horizontal="center", vertical="center")

            # Process test case data rows
            for r_idx, row_vals in enumerate(raw_rows_buffer, 1):
                name_val = str(row_vals[col_map.get("name", 0)]) if "name" in col_map and col_map["name"] < len(row_vals) and row_vals[col_map["name"]] else f"TC_{r_idx}"
                user_cmd = str(row_vals[col_map.get("user_command", 1)]) if "user_command" in col_map and col_map["user_command"] < len(row_vals) and row_vals[col_map["user_command"]] else ""
                actual_resp = str(row_vals[col_map.get("actual_resp", 3)]) if "actual_resp" in col_map and col_map["actual_resp"] < len(row_vals) and row_vals[col_map["actual_resp"]] else ""
                expected_resp = str(row_vals[col_map.get("expected_resp", 4)]) if "expected_resp" in col_map and col_map["expected_resp"] < len(row_vals) and row_vals[col_map["expected_resp"]] else ""

                if r_idx % 100 == 0:
                    gc.collect()

                res = self.evaluate_row_sync(
                    name=name_val,
                    user_cmd=user_cmd,
                    actual_resp=actual_resp,
                    expected_resp=expected_resp
                )

                status = res["auto_result"]
                total_eval_count += 1
                if status == "PASS":
                    pass_count += 1
                    fill = PASS_FILL
                elif status == "FAIL":
                    fail_count += 1
                    fill = FAIL_FILL
                else:
                    retest_count += 1
                    fill = RETEST_FILL

                # Build row values
                base_row = list(row_vals[:max_col_idx])
                # Pad if needed
                while len(base_row) < max_col_idx:
                    base_row.append("")

                eval_values = [res["auto_result"], res["score"], res["rule_info"], res["rca"], res["remediation"]]
                out_row_vals = base_row + eval_values
                ws_out.append(out_row_vals)

                curr_row = ws_out.max_row
                # Format status cell
                c1 = ws_out.cell(row=curr_row, column=col_start_idx)
                c1.fill = fill
                c1.alignment = Alignment(horizontal="center")

                # Overwrite pre-filled tool 'Result' column if present with AI verdict
                if "prev_result" in col_map and col_map["prev_result"] < len(base_row):
                    orig_res_cell = ws_out.cell(row=curr_row, column=col_map["prev_result"] + 1, value=res["auto_result"])
                    orig_res_cell.fill = fill
                    orig_res_cell.alignment = Alignment(horizontal="center")

        # Determine output filename
        if not output_excel_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_excel_path = str(input_path.parent / f"evaluated_{input_path.stem}_{timestamp}.xlsx")

        wb_out.save(output_excel_path)

        print("\n" + "=" * 60)
        print(" 🎉 AUTOMATED EVALUATION COMPLETE!")
        print("=" * 60)
        print(f" 📄 Total Test Cases Evaluated : {total_eval_count}")
        print(f" ✅ PASS                       : {pass_count}")
        print(f" ❌ FAIL                       : {fail_count}")
        print(f" ⚠️  RETEST                     : {retest_count}")
        if total_eval_count > 0:
            print(f" 📈 Pass Rate                  : {pass_count / total_eval_count * 100:.1f}%")
        print(f" 💾 Output Saved To            : {output_excel_path}")
        print("=" * 60 + "\n")

        return output_excel_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Excel Test Case Evaluator")
    parser.add_argument("--file", type=str, default="data/kết quảOM8NP.xlsx", help="Input Excel test case file")
    args = parser.parse_args()

    evaluator = ExcelTestEvaluator()
    evaluator.evaluate_file(args.file)
