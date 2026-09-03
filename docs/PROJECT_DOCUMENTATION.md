# ViVi Auto-Eval — Project Documentation

**Project**: RAG Chatbot & Automated Test Evaluation Engine for the VinFast ViVi voice assistant
**Scope of this document**: every module in the codebase, how they fit together, the evaluation algorithm in detail, the REST API, configuration, and known limitations.

---

## 1. What this project does

Two things live in one codebase:

1. **A RAG chatbot** ([src/rag_chain.py](../src/rag_chain.py)) that answers questions about VinFast vehicles using a ChromaDB vector store built from Owner Manuals, command-list spreadsheets, and other knowledge sources.
2. **An automated evaluation engine** that grades ViVi's real bench-test responses (recorded in Excel files: `User_command`, `Actual_resp`, `Expected_resp`, `Vivi_listen`) as `PASS`, `FAIL`, or `RETEST`, with a root-cause analysis (RCA) string, a similarity score, a severity tier, and an audit trace for every row.

The second half is the larger, more complex part of the codebase and the main subject of this document.

---

## 2. Directory layout

```
RAG-build-demo-1/
├── app.py                 # CLI entry point (chat + /eval-excel + /ingest commands)
├── gui.py                 # Desktop launcher: starts the FastAPI server + pywebview window
├── src/
│   ├── config.py           # Env vars, thresholds, embedding model factory
│   ├── rag_chain.py         # RAGChatbot class - retrieval + LLM answer generation
│   ├── ingest.py            # Loads data/ into ChromaDB (full + incremental)
│   ├── eval_router.py       # AdaptiveEvalRouter - the 5-tier grading engine
│   ├── eval_tools.py        # Similarity scoring, RAG/web search tools, health monitor
│   ├── eval_trace.py        # trace_id generation, severity classification, semantic diff
│   ├── agent_loop.py        # Self-correction loop for borderline scores (RAG + web retry)
│   ├── llm_judge.py         # Single-call local LLM escalation for unresolved RETEST cases
│   ├── embedding_trace.py   # Logs every embedding-model call for cost/perf monitoring
│   ├── excel_evaluator.py   # Batch Excel evaluation (CLI path, 16-worker pool)
│   ├── web_server.py        # FastAPI app - the desktop app's backend
│   ├── test_eval_agent.py   # LLM tool-calling agent for ad-hoc RCA queries
│   ├── agent_benchmark.py   # 108-case synthetic regression benchmark
│   └── crawl_vinfast_om.py  # Downloads Owner Manuals from VinFast's public OM API
├── static/                # Desktop app frontend (vanilla HTML/CSS/JS)
├── data/                   # Source documents + uploaded test files (see §7)
├── db/                     # ChromaDB persistent store (gitignored)
└── docs/                  # This file + the executive architecture doc
```

---

## 3. Core modules

### 3.1 `config.py`
Loads `.env` (`OPENAI_API_KEY`, `CHROMA_DB_DIR`, `MODEL_NAME`, `EMBEDDING_MODEL`), defines the four similarity thresholds used by the router (§5.3), and exposes `get_embedding_model()` — the single factory used everywhere an embedding model is needed. It picks OpenAI `text-embedding-3-small` if a real API key is present, otherwise falls back to the local, free HuggingFace `all-MiniLM-L6-v2` model. The returned model is wrapped in `TracedEmbeddings` (§3.9) before being cached.

### 3.2 `rag_chain.py`
`RAGChatbot` — retrieves top-k chunks from the shared ChromaDB collection, formats them into a context block, and asks an LLM (OpenAI → local HuggingFace Qwen2.5-1.5B → Ollama `qwen2.5:3b`, in that fallback order) to answer. If no LLM is reachable at all, it falls back to printing the raw retrieved chunks. Also exposes `get_vector_store()`, the single cached Chroma instance every other module queries.

### 3.3 `ingest.py`
Builds the vector store from `data/`:
- `.txt`/`.md`/`.pdf` files → chunked with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap), tagged `doc_type="owner_manual"` or `"knowledge"`.
- `.xlsx` files → each data row becomes one `Document` (not chunked further), tagged `doc_type="command_rule"`. Files that look like test-result exports (`is_test_result_file()` matches on filename substrings like `evaluated_`, `kết quả`, `testcase`) are skipped so evaluation outputs never pollute the knowledge base.
- Everything lands in **one single ChromaDB collection** at `config.CHROMA_DB_DIR` (`db/` by default) — there is no per-category separation.
- `ingest_single_file()` / `delete_single_file_vectors()` support incremental add/remove without rebuilding the whole DB (used by the web upload/delete endpoints).

### 3.4 `eval_router.py` — the grading engine
`AdaptiveEvalRouter.evaluate(name, user_cmd, actual_resp, expected_resp, testcase_category, vivi_listen)` is the single entry point. Full routing logic is in §5.

### 3.5 `eval_tools.py`
- `compute_similarity()` — grounds a bot answer against a (often long) reference spec. See §5.4 for the formula and its history.
- `compute_text_overlap_similarity()` — symmetric Dice-coefficient similarity for comparing two short, comparable-length texts (used only for the STT check, §5.2).
- `extract_relevant_sentence()` — picks the best-matching sentence/segment from a long text for **display** in the `Matched_Rule_Spec` column (keyword-count heuristic; not used for scoring — see §8.2 for why that matters).
- `rag_rule_search()` / `rag_spec_search()` — query the shared Chroma collection, filtered by `doc_type` metadata where possible, with in-memory caching (`_RULE_SEARCH_CACHE`).
- `web_search_verification()` — DuckDuckGo live search with an in-memory result cache (`_WEB_SEARCH_CACHE`) and a 2-second timeout.
- `BatchHealthMonitor` — a sliding-window circuit breaker: if the web-search error rate exceeds 15% over the last 50 calls, it opens the circuit and short-circuits further web calls (logged as `ALERT: Web search error rate ...`) until `.reset()`.
- `eval_test_result()` / `compare_expected_actual()` / `generate_eval_report()` — generic tool-calling primitives used by `test_eval_agent.py`, independent of the main router.

### 3.6 `eval_trace.py`
- `generate_trace_log()` — builds the full audit record for a row: `trace_id`, `system_version`, `severity`, `semantic_error_pct`, `error_category`, `semantic_diff`, `resolved_by`, and the raw retrieved chunks. This is what powers the "Trace Log" modal in the dashboard.
- `classify_severity(auto_result, sim_score, is_stt_mismatch, is_false_refusal)` — maps a verdict + score into `PASS` / `LOW` / `MEDIUM` / `HIGH`:
  | Condition | Severity |
  |---|---|
  | `auto_result == PASS` and `sim_score >= 0.85` | PASS |
  | `is_stt_mismatch` | LOW (acoustic issue, not a logic bug) |
  | `is_false_refusal` or `sim_score < 0.40` | HIGH |
  | `sim_score < 0.70` | MEDIUM |
  | else | LOW |
- `compute_semantic_diff()` — word-level diff between expected and actual, used to append a `[Semantic Diff Failure: missing key terms ...]` note to the RCA. **Known limitation**: it has no concept of "critical" vs. "filler" words, so it sometimes flags greeting/header words (e.g. "giới", "thiệu", "chúc", "mừng") as "missing critical terms" when the expected text is a generic section intro — see §8.2.
- `ErrorCategory` — `FACT_HALLUCINATION`, `FALSE_REFUSAL`, `STT_ACOUSTIC_MISMATCH`, `COMPLETENESS_LOSS`, `INFRASTRUCTURE_OUTAGE`, `NONE`.

### 3.7 `agent_loop.py`
`AgentEvalLoop.run_correction_loop(user_cmd, actual_resp, initial_score)` — for scores in the borderline band (0.25–0.45), tries two escalation steps before giving up:
1. RAG query expansion (`"{user_cmd} thông số tính năng vận hành hướng dẫn"`) against the vector store.
2. Live web search, if step 1 doesn't clear `RAG_PASS_THRESHOLD`.

If either step's result scores above the relevant threshold, it returns a resolved `PASS`; otherwise it reports `resolved: False` and the caller falls back to `RETEST`.

### 3.8 `llm_judge.py`
Single-call local LLM (`qwen2.5:3b` via Ollama, `http://127.0.0.1:11434`) escalation for RETEST cases the keyword scorer can't resolve. See §5.5 for why this replaced an earlier multi-agent debate design.

### 3.9 `embedding_trace.py`
`TracedEmbeddings` wraps whatever embedding model `config.get_embedding_model()` returns, logging every `embed_documents()`/`embed_query()` call (timestamp, provider, model, text count, latency, success/error) into an in-memory ring buffer (max 2000 entries). Exposed via `GET /api/embedding/trace`. Purely for monitoring — does not affect scoring.

### 3.10 `excel_evaluator.py`
`ExcelTestEvaluator.evaluate_file()` — the CLI/batch path:
1. Streams the input workbook sheet by sheet, auto-detects the header row and column mapping via `_detect_columns()` (handles metadata blocks before the real header, and multiple naming conventions per column — see §7.1).
2. Runs all data rows through `router.evaluate()` in a **16-worker `ThreadPoolExecutor`**.
3. Writes an output workbook with 5 new columns appended: `Auto_Eval_Result`, `Similarity_Score(%)`, `Matched_Rule_Spec`, `Root_Cause_Analysis`, `Suggested_Remediation`, with conditional cell coloring (green/red/yellow).

`detect_testcase_category()` classifies a sheet as `owner_manual` / `command_rule` / `general_knowledge` / `error_code` / `general` from the filename+sheet name, which steers which vector-store filter the router prefers.

### 3.11 `web_server.py`
FastAPI app (see §6 for the full endpoint table). `run_batch_evaluation_task()` is the web-upload equivalent of `excel_evaluator.evaluate_file()` — **note**: unlike the CLI path, it currently processes rows in a plain sequential loop, not the 16-worker pool (a known inconsistency, see §8.1).

### 3.12 `test_eval_agent.py`
`TestEvalAgent` — a small ReAct-style tool-calling loop (`TOOL_CALL: <name> | {...}` text protocol) used by the `/eval` CLI command and `/api/eval/single`-adjacent flows for free-form "evaluate this test result and explain why" queries. Independent of `AdaptiveEvalRouter`.

### 3.13 `agent_benchmark.py`
`get_real_ground_truth_testset()` returns 108 hand-written, stratified synthetic test cases across 6 categories (Explicit Spec, Domain RAG, Borderline, STT Mismatch, False Refusal, Tier 5 Conversational). `AgentBenchmarkSuite.run_benchmark()` runs them through the router and reports accuracy, a confusion matrix, false-pass/false-reject rates, and per-slice breakdowns. This is a synthetic regression check, not a substitute for evaluating real production data (see §8.3).

### 3.14 `crawl_vinfast_om.py`
Standalone script that pulls Owner Manuals from VinFast's public OM API (`omapi.vinfastauto.com`) for a hardcoded list of models (VF3/5/6/7/8/8NP/9, VFe34, Fadil, LuxA/SA 2.0), strips the HTML, and writes `data/om_manuals/VinFast_OM_{model}_{version}_{lang}.txt`. Not wired into the main app — run manually when you need fresh manuals.

---

## 4. Frontend (`static/`)

Vanilla HTML/CSS/JS single-page dashboard with four tabs: **Batch Evaluation** (upload → progress bar → results table with PASS/FAIL/RETEST filters, a trace-log modal per row, Excel download), **RAG Knowledge Assistant** (chat UI hitting `/api/chat`), **Single RCA Workbench** (`/api/eval/single`), and **Data Sources** (list/upload/delete files, hits `/api/data/*`). All dynamic content is escaped via a shared `escapeHtml()` before insertion into the DOM — no XSS risk from uploaded test data.

---

## 5. The evaluation pipeline in detail

`AdaptiveEvalRouter.evaluate()` runs each row through, in order:

### 5.1 Pre-checks (before tier routing)
- **Empty/near-empty actual response** → `RETEST`, score 0 (or `FAIL` if it's a sensitive/policy query with no refusal — see 5.3 Tier 1).
- **Truncated response** (< 25 chars, not a refusal phrase) → `FAIL`, score 20.
- **Refusal on a non-sensitive query** (matches a hardcoded list of Vietnamese refusal phrases in `is_true_refusal()`) → `FAIL`, tagged `is_false_refusal`.

### 5.2 STT hearing-mismatch check
Compares `Vivi_listen` (what the STT engine transcribed) against `User_command` using `compute_text_overlap_similarity()` — a **symmetric** Dice-coefficient measure, distinct from the main scorer because both texts here are short and comparably sized (no length asymmetry to correct for). If the transcript is a bare wake-word (`"hey vinfast"`, `"vivi"`, etc.) or the overlap score is below 0.70 for a command longer than 20 characters, the row is short-circuited to `RETEST` with an `STT_ACOUSTIC_MISMATCH` error category — the bot isn't judged on content it never actually heard.

### 5.3 The 5 tiers
| Tier | Trigger | Pass condition |
|---|---|---|
| 1. Policy/Refusal | Query matches sensitive/political/opinion regex patterns | Refusal or non-answer → `PASS` |
| 2. Explicit Spec | `Expected_resp` column has real content (> 10 chars) | `compute_similarity(actual, expected) >= 0.45` |
| 3. Domain RAG | Domain keyword hit or RAG returns chunks | `compute_similarity(actual, top_chunk) >= 0.45` |
| 4. Web Fact Verification | Query matches factual/temporal patterns | `compute_similarity(actual, web_snippets) >= 0.50` |
| 5. General Conversational | Nothing else matched | Resolved via `AgentEvalLoop`, else `RETEST` |

Tiers 2–4 share the same borderline-band logic: a score in **[0.25, 0.45)** first tries `AgentEvalLoop.run_correction_loop()` (RAG/web re-search); if that doesn't resolve it, it now returns an explicit `RETEST` **with the real computed score** (not a placeholder). A score **below 0.25** returns an explicit `FAIL`. *(This explicit-return behavior was added mid-project — earlier versions of Tiers 3/4 had no `else` branch and silently fell through to the Tier 5 fallback, which rewrote the score to a flat 50.0 regardless of how wrong the answer actually was. See §8's changelog.)*

### 5.4 `compute_similarity()` — the core scoring formula

```python
words_act = set(non_stopword_tokens(actual))
words_exp = set(non_stopword_tokens(expected))
matched   = words_act & words_exp

precision       = len(matched) / len(words_act)
coverage_factor = min(1.0, len(matched) / 12.0)
boosted         = min(1.0, precision * 1.15) if precision >= 0.4 else precision

score = boosted * coverage_factor
```

Why this specific shape (calibrated against ~1,100 real production rows):
- **Precision, not recall, over `expected`**: `expected` is frequently an entire retrieved Owner-Manual chunk covering more ground than the specific question, so requiring the bot to "recall" all of it would unfairly punish short, correct, on-point answers (verified empirically — a full precision+recall F1 formula was tried and it collapsed the PASS rate from 65% to 9%).
- **`coverage_factor` (the matched-word-count floor)**: precision alone is gameable — a short, completely irrelevant response (e.g. a canned "please connect to internet" decline) can share just 4–5 common words with a long reference and hit >90% precision. The floor requires enough *absolute* overlap before precision is trusted at face value. `/12.0` was chosen by sweeping thresholds against a labeled set of confirmed-bad (decline-style false positive) vs. confirmed-good real answers; it caught ~97% of the bad cases while only misclassifying ~2.5% of the good ones.
- Special-cased shortcuts before the formula: exact string match → `1.0`; a small hardcoded list of pure filler/canned phrases (`"dạ em đây"`, `"cần em hỗ trợ gì"`, etc.) on a short response against a long reference → `0.1` regardless of accidental word overlap.

### 5.5 LLM Judge escalation for RETEST
For any `RETEST` verdict that (a) isn't an STT mismatch, (b) isn't an empty-response case, and (c) has real reference content to compare against, `_build_result()` makes one call to a local Ollama model asking it to judge PASS/FAIL directly. If resolved, it overrides the verdict (`score = max(score, 80.0)` on a judge PASS; the original low score is kept on a judge FAIL, since it already reflects a real mismatch). If the call errors, times out, or the output can't be parsed, the original `RETEST` is kept unchanged — it never fabricates a verdict.

**Why single-call instead of multi-agent debate**: a 2–3 call design (an "Advocate" arguing the answer is correct, a "Skeptic" arguing it isn't, optionally + a synthesis Judge) was built and benchmarked first. With the small local models available (`qwen2.5:3b`, `llama3.2:latest`, 3B-parameter class, CPU-only inference), the Advocate role — instructed to defend the answer even when it shouldn't be defensible — reliably talked the process into false `PASS` verdicts on cases a plain single-judge call correctly caught (generic canned-greeting non-answers, visibly truncated responses). Manual spot-checks of disagreements between the two designs found the single-call judge correct in all checked cases where the debate was wrong, while being **5–6× faster** (one ~4.6s call vs. 25–45s for the multi-call chain) and structurally simpler. The debate module (`debate_judge.py`) was deleted after this comparison.

---

## 6. REST API (`web_server.py`, mounted under `/`)

| Method & Path | Purpose |
|---|---|
| `GET /api/embedding/trace?limit=100` | Embedding-call monitoring summary + recent log entries |
| `GET /api/data/categories` | List data subfolder categories |
| `GET /api/data/sources` | Inventory of embedded files with chunk counts (queries ChromaDB metadata) |
| `POST /api/data/upload` | Upload a file to `data/<category>/`, triggers incremental re-ingestion |
| `POST /api/data/delete` | Delete a source file + purge its vectors |
| `POST /api/eval/upload` | Upload an `.xlsx` test file, starts a background batch evaluation task |
| `GET /api/eval/progress/{task_id}` | Poll batch task status/progress/results |
| `GET /api/eval/download/{filename}` | Download the evaluated output workbook |
| `POST /api/chat` | Ask the RAG chatbot a question |
| `POST /api/eval/single` | Evaluate one test case ad-hoc through `AdaptiveEvalRouter` |

CORS is currently `allow_origins=["*"]` combined with `allow_credentials=True`, which is technically an invalid combination per the CORS spec (low risk while bound to `127.0.0.1`, worth tightening if ever exposed beyond localhost).

---

## 7. Data conventions

### 7.1 Expected Excel columns (auto-detected, case/spacing-insensitive, Vietnamese or English)
`Name_testcase`, `User_command`, `Vivi_listen`, `Actual_resp`, `Expected_resp`, `Result` (previous human label, preserved but not read as input), `Latency`. Header row is located by scanning for these patterns, so metadata/title blocks above the real header (common in real bench exports) are handled automatically.

### 7.2 What NOT to put in `data/`
Files whose name contains `evaluated_`, `kết quả`, `ket qua`, `testcase`, or `result_` are treated as **test result logs**, not knowledge, and are excluded from ingestion (`is_test_result_file()`). Files that don't match these patterns but are still test logs (e.g. a file literally named `test_fix_thuongthuc.xlsx`) will slip through and pollute the RAG index with irrelevant content — see §8.4.

---

## 8. Known limitations & fix history

This section exists because the codebase had several real, confirmed bugs found and fixed during development — worth keeping visible so they aren't silently reintroduced.

### 8.1 Batch API path doesn't parallelize
`web_server.py::run_batch_evaluation_task()` (the web-upload path) processes rows in a plain `for` loop, unlike `excel_evaluator.py`'s 16-worker pool (the CLI path). Not yet fixed — batch evaluations triggered from the dashboard are slower than the same file run via CLI.

### 8.2 `compute_semantic_diff()` has no "generic word" awareness
When `expected_resp` is a big generic section intro (e.g. an OM manual's "Introduction" boilerplate shared across many unrelated questions in that section), the diff can flag greeting/header words ("giới", "thiệu", "chúc", "mừng") as "missing critical key terms," which is misleading in the RCA even when the underlying score/verdict is otherwise reasonable. Not fixed — cosmetic (RCA text quality), doesn't affect the verdict.

### 8.3 Synthetic benchmark ≠ real accuracy
`agent_benchmark.py`'s 108 hand-written cases gave ~74% accuracy at one point, but a full run against ~1,800–2,500 real production rows (with genuine human-graded labels) showed the *algorithm* was actually being penalized by **label noise in the source file**, not code bugs — e.g. rows where a canned "please connect to internet" decline was graded `PASS` by a human reviewer were later confirmed by the project owner to be *human labeling mistakes*, not intended behavior. Always validate against real data with a human in the loop before trusting either number in isolation.

### 8.4 Vector DB can be polluted by unrecognized test-log files
See §7.2. `data/test_fix_thuongthuc.xlsx` and `data/test_stt_mismatch_eval.xlsx` are known instances that were ingested as if they were domain knowledge, causing unrelated RAG search results (e.g. a tire-pressure question returning content about a forest in An Giang). Not fixed — either broaden `is_test_result_file()`'s pattern list or move these files out of `data/`.

### 8.5 Fixed during this project's evaluation-quality pass (for reference)
- **Tier 3/4 silent fallthrough**: a low-confidence Domain-RAG or Web-Fact score used to fall through to the Tier 5 fallback and get rewritten to a flat `RETEST @ 50.0`, hiding genuinely bad answers behind a manufactured "medium confidence" number. Now returns explicit `FAIL`/`RETEST` with the real score.
- **STT check reused the wrong similarity function**: `compute_similarity()` (tuned for grounding a long answer against a long reference) was also used to compare `Vivi_listen` against `User_command` — two short, comparable-length texts — where its coverage-factor floor unfairly penalized correctly-heard commands under ~12 content words, mislabeling them as acoustic mismatches. Split into a dedicated `compute_text_overlap_similarity()`.
- **`compute_similarity()` precision-only formula was exploitable**: an early version scored pure keyword precision with an aggressive boost (`recall_exp * 1.5 + 0.3`), letting short, completely irrelevant decline messages hit 90%+ scores purely by sharing a handful of generic connector words with a long reference. Replaced with the matched-word-count-gated formula in §5.4.
- **Dead code removed**: `src/ingest_rules.py` (referenced two `config` attributes that didn't exist — would crash if run, and nothing imported it) and duplicate `normalize_text`/`compute_similarity`/`should_trigger_web_search` copies inside `excel_evaluator.py` that nothing called.

---

## 9. Configuration reference (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (empty) | If unset/placeholder, everything falls back to free local models (HuggingFace embeddings, Ollama LLM) |
| `CHROMA_DB_DIR` | `db` | Vector store location, relative to project root |
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI chat model, when a key is configured |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model, when a key is configured |

Similarity thresholds (`src/config.py`, not env-configurable): `EXPLICIT_SPEC_PASS_THRESHOLD = RAG_PASS_THRESHOLD = 0.45`, `WEB_PASS_THRESHOLD = 0.50`, `BORDERLINE_LOW_THRESHOLD = 0.25`.

---

## 10. Running it

```bash
# CLI (chat + /eval-excel + /ingest)
python app.py

# Desktop app (FastAPI + pywebview, falls back to opening a browser tab)
python gui.py

# Just the API server
python -m uvicorn src.web_server:app --host 127.0.0.1 --port 8000

# Rebuild the vector DB from data/
python -c "from src.ingest import ingest_documents; ingest_documents()"

# Synthetic regression benchmark
python -m src.agent_benchmark
```

Local LLM fallback requires `ollama serve` running with `qwen2.5:3b` pulled (`ollama pull qwen2.5:3b`) — used by `llm_judge.py` and as the last-resort chat LLM fallback in `rag_chain.py`/`test_eval_agent.py`.
