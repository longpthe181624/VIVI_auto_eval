/* ===================================================================
   VIVI AUTO-EVAL DESKTOP APPLICATION TOOL - CLIENT LOGIC
   =================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  initTabNavigation();
  initBatchEvaluation();
  initRAGChat();
  initSingleRCA();
  initDataManager();
});

// -------------------------------------------------------------------
// 1. TAB NAVIGATION LOGIC
// -------------------------------------------------------------------
function initTabNavigation() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabBtns.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(targetId).classList.add("active");
    });
  });
}

// -------------------------------------------------------------------
// 2. BATCH EXCEL EVALUATION HUB LOGIC
// -------------------------------------------------------------------
function initBatchEvaluation() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("excel-file-input");
  const btnBrowse = document.getElementById("btn-browse-excel");
  const fileNameDisplay = document.getElementById("file-name-display");
  const btnStartEval = document.getElementById("btn-start-eval");

  const progressPanel = document.getElementById("progress-panel");
  const progressStatusText = document.getElementById("progress-status-text");
  const progressPctText = document.getElementById("progress-pct-text");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const progressCountText = document.getElementById("progress-count-text");
  const progressRateText = document.getElementById("progress-rate-text");

  const statsGrid = document.getElementById("stats-grid");
  const statTotal = document.getElementById("stat-total");
  const statPassRate = document.getElementById("stat-pass-rate");
  const statFail = document.getElementById("stat-fail");
  const statRetest = document.getElementById("stat-retest");

  const resultsPanel = document.getElementById("results-panel");
  const resultsTableBody = document.getElementById("results-table-body");
  const btnDownloadReport = document.getElementById("btn-download-report");
  const searchInput = document.getElementById("eval-search-input");
  const filterBtns = document.querySelectorAll(".filter-btn");

  let selectedFile = null;
  let currentTaskId = null;
  let pollInterval = null;
  let allResults = [];
  let activeFilter = "ALL";
  let latestOutputFile = "";

  btnBrowse.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.name.endsWith(".xlsx")) {
      alert("Please select a valid Excel workbook (.xlsx).");
      return;
    }
    selectedFile = file;
    fileNameDisplay.textContent = `Selected File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    btnStartEval.disabled = false;
  }

  btnStartEval.addEventListener("click", async () => {
    if (!selectedFile) return;

    btnStartEval.disabled = true;
    progressPanel.classList.remove("hidden");
    statsGrid.classList.remove("hidden");
    resultsPanel.classList.add("hidden");

    progressStatusText.textContent = "Uploading Excel test suite...";
    progressPctText.textContent = "0.0%";
    progressBarFill.style.width = "0%";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const resp = await fetch("/api/eval/upload", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json();
        alert(`Failed to start evaluation: ${err.detail || "Upload error"}`);
        btnStartEval.disabled = false;
        return;
      }

      const data = await resp.json();
      currentTaskId = data.task_id;
      startProgressPolling();

    } catch (e) {
      alert(`Network error starting evaluation: ${e.message}`);
      btnStartEval.disabled = false;
    }
  });

  function startProgressPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      if (!currentTaskId) return;

      try {
        const resp = await fetch(`/api/eval/progress/${currentTaskId}`);
        if (!resp.ok) return;

        const task = await resp.json();
        updateTelemetryUI(task);

        if (task.status === "COMPLETED" || task.status === "FAILED") {
          clearInterval(pollInterval);
          btnStartEval.disabled = false;

          if (task.status === "COMPLETED") {
            progressStatusText.textContent = "Evaluation Run Finished Successfully!";
            allResults = task.results || [];
            latestOutputFile = task.output_file || "";
            resultsPanel.classList.remove("hidden");
            renderResultsTable();
          } else {
            progressStatusText.textContent = `Evaluation Failed: ${task.error || "Unknown error"}`;
          }
        }
      } catch (e) {
        console.error("Progress polling error:", e);
      }
    }, 1000);
  }

  function updateTelemetryUI(task) {
    const pct = task.progress_pct || 0;
    progressPctText.textContent = `${pct.toFixed(1)}%`;
    progressBarFill.style.width = `${pct}%`;

    progressStatusText.textContent = `Evaluating Test Cases (${task.evaluated_rows || 0} / ${task.total_rows || 0})...`;
    progressCountText.textContent = `Processed: ${task.evaluated_rows || 0} / ${task.total_rows || 0} rows`;
    progressRateText.textContent = `Rate: ${task.rows_per_sec || 0.0} rows/sec`;

    const total = task.evaluated_rows || 0;
    const pass = task.pass_count || 0;
    const passRate = total > 0 ? ((pass / total) * 100).toFixed(1) : "0.0";

    statTotal.textContent = total;
    statPassRate.textContent = `${passRate}%`;
    statFail.textContent = task.fail_count || 0;
    statRetest.textContent = task.retest_count || 0;
  }

  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      filterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeFilter = btn.getAttribute("data-filter");
      renderResultsTable();
    });
  });

  searchInput.addEventListener("input", () => renderResultsTable());

  btnDownloadReport.addEventListener("click", () => {
    if (latestOutputFile) {
      window.location.href = `/api/eval/download/${latestOutputFile}`;
    }
  });

  function renderResultsTable() {
    resultsTableBody.innerHTML = "";
    const searchTerm = searchInput.value.toLowerCase().trim();

    const filtered = allResults.filter((item) => {
      const matchesFilter = activeFilter === "ALL" || item.status === activeFilter;
      const matchesSearch =
        !searchTerm ||
        item.id.toLowerCase().includes(searchTerm) ||
        item.user_command.toLowerCase().includes(searchTerm) ||
        item.actual_resp.toLowerCase().includes(searchTerm) ||
        item.rule_info.toLowerCase().includes(searchTerm);

      return matchesFilter && matchesSearch;
    });

    if (filtered.length === 0) {
      resultsTableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No evaluation results match current filter.</td></tr>`;
      return;
    }

    filtered.forEach((row) => {
      const tr = document.createElement("tr");
      const badgeClass =
        row.status === "PASS"
          ? "badge-pass"
          : row.status === "FAIL"
          ? "badge-fail"
          : "badge-retest";

      tr.innerHTML = `
        <td><strong>${escapeHtml(row.id)}</strong></td>
        <td><span class="badge ${badgeClass}">${row.status}</span></td>
        <td>${row.score}%</td>
        <td>${escapeHtml(row.user_command)}</td>
        <td>${escapeHtml(row.actual_resp)}</td>
        <td>${escapeHtml(row.rule_info)}</td>
        <td>${escapeHtml(row.rca)}</td>
      `;
      resultsTableBody.appendChild(tr);
    });
  }
}

// -------------------------------------------------------------------
// 3. RAG KNOWLEDGE ASSISTANT LOGIC
// -------------------------------------------------------------------
function initRAGChat() {
  const chatMessages = document.getElementById("chat-messages-container");
  const chatInput = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-send-chat");
  const modelSelect = document.getElementById("chat-model-select");
  const langSelect = document.getElementById("chat-lang-select");

  btnSend.addEventListener("click", () => sendChatMessage());
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  async function sendChatMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    chatInput.value = "";
    appendChatBubble(query, "user");

    const botBubble = appendChatBubble("Searching VinFast vehicle knowledge base...", "bot");

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          car_model: modelSelect.value || null,
          lang: langSelect.value || null,
        }),
      });

      if (!resp.ok) {
        botBubble.innerHTML = "Error processing query. Ensure server is active.";
        return;
      }

      const data = await resp.json();
      botBubble.innerHTML = escapeHtml(data.answer);

      if (data.sources && data.sources.length > 0) {
        const sourcesDiv = document.createElement("div");
        sourcesDiv.className = "chat-sources";
        sourcesDiv.innerHTML = `<strong>Referenced Sources:</strong><br>` + data.sources.map((s) => `• ${escapeHtml(s)}`).join("<br>");
        botBubble.appendChild(sourcesDiv);
      }
    } catch (e) {
      botBubble.innerHTML = `Network error: ${e.message}`;
    }
  }

  function appendChatBubble(text, sender) {
    const div = document.createElement("div");
    div.className = `chat-bubble ${sender}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
  }
}

// -------------------------------------------------------------------
// 4. SINGLE TEST CASE RCA WORKBENCH LOGIC
// -------------------------------------------------------------------
function initSingleRCA() {
  const form = document.getElementById("single-eval-form");
  const resultPanel = document.getElementById("single-result-content");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultPanel.innerHTML = "<p class='placeholder-text'>Executing adaptive evaluation and root cause analysis...</p>";

    const payload = {
      name: document.getElementById("single-test-id").value.trim() || "TC_Single",
      user_cmd: document.getElementById("single-user-cmd").value.trim(),
      actual_resp: document.getElementById("single-actual-resp").value.trim(),
      expected_resp: document.getElementById("single-expected-resp").value.trim(),
    };

    try {
      const resp = await fetch("/api/eval/single", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        resultPanel.innerHTML = "<p class='placeholder-text' style='color: var(--color-fail);'>Error running single evaluation.</p>";
        return;
      }

      const res = await resp.json();
      const badgeClass =
        res.auto_result === "PASS"
          ? "badge-pass"
          : res.auto_result === "FAIL"
          ? "badge-fail"
          : "badge-retest";

      resultPanel.innerHTML = `
        <div style="margin-bottom: 16px;">
          <span class="badge ${badgeClass}" style="font-size: 14px; padding: 4px 12px;">${res.auto_result}</span>
          <span style="font-weight: 700; margin-left: 12px; font-size: 16px;">Score: ${res.score}%</span>
        </div>
        <div class="form-group">
          <label>Matched Rule / Spec Reference</label>
          <div class="form-control" style="background: rgba(0,0,0,0.4);">${escapeHtml(res.rule_info)}</div>
        </div>
        <div class="form-group">
          <label>Root Cause Analysis (RCA)</label>
          <div class="form-control" style="background: rgba(0,0,0,0.4);">${escapeHtml(res.rca)}</div>
        </div>
        <div class="form-group">
          <label>Suggested Remediation Plan</label>
          <div class="form-control" style="background: rgba(0,0,0,0.4);">${escapeHtml(res.remediation)}</div>
        </div>
      `;
    } catch (e) {
      resultPanel.innerHTML = `<p class="placeholder-text" style="color: var(--color-fail);">Network error: ${e.message}</p>`;
    }
  });
}

// -------------------------------------------------------------------
// 5. KNOWLEDGE DATA SOURCE & CATEGORY MANAGER LOGIC
// -------------------------------------------------------------------
function initDataManager() {
  const categorySelect = document.getElementById("data-category-select");
  const uploadForm = document.getElementById("data-upload-form");
  const fileInput = document.getElementById("data-file-input");
  const btnUpload = document.getElementById("btn-upload-source");
  const sourcesTableBody = document.getElementById("sources-table-body");
  const inventoryCountText = document.getElementById("inventory-count-text");
  const btnRefresh = document.getElementById("btn-refresh-sources");

  loadCategories();
  loadEmbeddedSources();

  btnRefresh.addEventListener("click", () => {
    loadCategories();
    loadEmbeddedSources();
  });

  async function loadCategories() {
    try {
      const resp = await fetch("/api/data/categories");
      if (!resp.ok) return;
      const data = await resp.json();

      categorySelect.innerHTML = "";
      data.categories.forEach((cat) => {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.textContent = cat;
        categorySelect.appendChild(opt);
      });
    } catch (e) {
      console.error("Error loading categories:", e);
    }
  }

  async function loadEmbeddedSources() {
    sourcesTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Loading embedded sources inventory...</td></tr>`;

    try {
      const resp = await fetch("/api/data/sources");
      if (!resp.ok) {
        sourcesTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--color-fail);">Error loading data sources.</td></tr>`;
        return;
      }

      const data = await resp.json();
      const sources = data.sources || [];

      let totalChunks = 0;
      sources.forEach((s) => (totalChunks += s.chunks || 0));
      inventoryCountText.textContent = `Total Embedded Vector Chunks: ${totalChunks.toLocaleString()}`;

      if (sources.length === 0) {
        sourcesTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No embedded data sources found in vector database.</td></tr>`;
        return;
      }

      sourcesTableBody.innerHTML = "";
      sources.forEach((item) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${escapeHtml(item.category)}</strong></td>
          <td>${escapeHtml(item.source_name)}</td>
          <td>${item.chunks.toLocaleString()}</td>
          <td><span class="badge badge-pass">${item.status}</span></td>
          <td>
            <button class="btn btn-danger btn-delete-source" data-name="${escapeHtml(item.source_name)}" data-category="${escapeHtml(item.category)}">Delete</button>
          </td>
        `;
        sourcesTableBody.appendChild(tr);
      });

      // Attach delete button listeners
      document.querySelectorAll(".btn-delete-source").forEach((btn) => {
        btn.addEventListener("click", () => {
          const srcName = btn.getAttribute("data-name");
          const cat = btn.getAttribute("data-category");
          deleteSourceData(srcName, cat);
        });
      });
    } catch (e) {
      sourcesTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--color-fail);">Network error: ${e.message}</td></tr>`;
    }
  }

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    btnUpload.disabled = true;
    btnUpload.textContent = "Uploading & Indexing...";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("category", categorySelect.value);

    try {
      const resp = await fetch("/api/data/upload", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json();
        alert(`Upload error: ${err.detail || "Failed to upload file"}`);
      } else {
        const res = await resp.json();
        alert(res.message);
        fileInput.value = "";
        setTimeout(loadEmbeddedSources, 2000);
      }
    } catch (e) {
      alert(`Network error uploading file: ${e.message}`);
    } finally {
      btnUpload.disabled = false;
      btnUpload.textContent = "Upload & Index File";
    }
  });

  async function deleteSourceData(sourceName, category) {
    if (!confirm(`Are you sure you want to delete source file '${sourceName}' from category '${category}' and purge its vector embeddings?`)) {
      return;
    }

    try {
      const resp = await fetch("/api/data/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_name: sourceName, category: category }),
      });

      if (!resp.ok) {
        alert("Failed to delete source data.");
        return;
      }

      const res = await resp.json();
      alert(res.message);
      setTimeout(loadEmbeddedSources, 2000);
    } catch (e) {
      alert(`Network error deleting source: ${e.message}`);
    }
  }
}

// Helper HTML Escaper
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
