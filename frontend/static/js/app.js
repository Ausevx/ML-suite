// ML-Suite Frontend Application
// Main JavaScript file handling all module interactions

// Global state
const appState = {
  currentModule: "ocr",
  currentDataset: null,
  currentModel: null,
  consoleLogs: [],
  currentConsoleFilter: "all",
  statsInterval: null,
  droppedColumns: new Set(),
  settings: {
    scanlines: false,
    flicker: false,
    glow: false,
    vignette: true,
    systemStats: true,
    statsFrequency: 2000,
    theme: "amber",
    crtCurvature: false,
    crtBrightness: 100,
    gpuEnabled: true,
    gpuForceCpu: false,
    gpuPreferredBackend: "auto",
    showMetrics: {
      r2Score: true,
      rmse: true,
      mae: true,
      mse: true,
      adjustedR2: false,
      residuals: true,
    },
    showCharts: {
      actualVsPredicted: true,
      residualPlot: true,
      featureImportance: true,
      distribution: false,
      learningCurve: false,
      correlationMatrix: false,
      confusionMatrix: true,
      rocCurve: true,
      clusterScatter: true,
      screePlot: true,
      transformedScatter: true,
    },
  },
};

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  applySettings();
  initializeNavigation();
  initializeSystemStats();
  initializeConsoleModule();
  initializeOCRModule();
  initializeTrainerModule();
  initializeModelsModule();
  initializeNotebookModule();
  initializeSettingsModule();

  // Log initial message
  logToConsole("ML-Suite initialized successfully", "info");
});

// Navigation
function initializeNavigation() {
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const module = link.getAttribute("data-module");
      switchModule(module);
    });
  });
}

function switchModule(moduleName) {
  // Update navigation
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.remove("active");
  });
  document
    .querySelector(`[data-module="${moduleName}"]`)
    .classList.add("active");

  // Update modules
  document.querySelectorAll(".module").forEach((mod) => {
    mod.classList.remove("active");
  });

  const moduleMap = {
    ocr: "ocr-module",
    trainer: "trainer-module",
    models: "models-module",
    notebook: "notebook-module",
    console: "console-module",
    settings: "settings-module",
  };

  document.getElementById(moduleMap[moduleName]).classList.add("active");
  appState.currentModule = moduleName;

  // Load module-specific data
  if (moduleName === "models") {
    loadModels();
  }

  logToConsole(`Switched to ${moduleName} module`, "info");
}

// System Stats Module
function initializeSystemStats() {
  if (appState.settings.systemStats) {
    updateSystemStats();
    appState.statsInterval = setInterval(
      updateSystemStats,
      appState.settings.statsFrequency
    );
  }
}

function restartSystemStats() {
  // Clear existing interval
  if (appState.statsInterval) {
    clearInterval(appState.statsInterval);
  }

  // Restart with new settings
  if (appState.settings.systemStats) {
    updateSystemStats();
    appState.statsInterval = setInterval(
      updateSystemStats,
      appState.settings.statsFrequency
    );
  }
}

async function updateSystemStats() {
  if (!appState.settings.systemStats) return;

  try {
    const response = await fetch("/api/system/stats");
    const data = await response.json();

    if (data.error) {
      document.getElementById("system-status").textContent = "ERROR";
      return;
    }

    // Update CPU
    document.getElementById(
      "cpu-usage"
    ).textContent = `${data.cpu.percent.toFixed(1)}%`;
    document.getElementById("cpu-bar").style.width = `${data.cpu.percent}%`;

    // Update RAM
    document.getElementById(
      "ram-usage"
    ).textContent = `${data.ram.percent.toFixed(1)}%`;
    document.getElementById("ram-bar").style.width = `${data.ram.percent}%`;

    // Update GPU if available
    const gpuUsageElement = document.getElementById("gpu-usage");
    const gpuBarElement = document.getElementById("gpu-bar");
    const gpuSection = document.getElementById("gpu-section");
    const gpuBarContainer = document.getElementById("gpu-bar-container");

    if (data.gpu && data.gpu.available && gpuUsageElement && gpuBarElement) {
      const gpuUtil = data.gpu.utilization || 0;
      gpuUsageElement.textContent = `${gpuUtil.toFixed(1)}%`;
      gpuBarElement.style.width = `${gpuUtil}%`;

      // Show GPU section
      if (gpuSection) gpuSection.style.display = "flex";
      if (gpuBarContainer) gpuBarContainer.style.display = "block";
    } else {
      // Hide GPU section if not available
      if (gpuSection) gpuSection.style.display = "none";
      if (gpuBarContainer) gpuBarContainer.style.display = "none";
    }

    // Update status
    document.getElementById("system-status").textContent =
      data.status.toUpperCase();
  } catch (error) {
    console.error("Failed to fetch system stats:", error);
  }
}

// Console/Logs Module
function initializeConsoleModule() {
  // Tab switching
  const tabs = document.querySelectorAll(".console-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      appState.currentConsoleFilter = tab.getAttribute("data-tab");
      renderConsoleLogs();
    });
  });

  // Clear console button
  document.getElementById("clear-console-btn").addEventListener("click", () => {
    appState.consoleLogs = [];
    renderConsoleLogs();
    logToConsole("Console cleared", "info");
  });

  // Export logs button
  document
    .getElementById("export-logs-btn")
    .addEventListener("click", exportLogs);
}

function logToConsole(message, type = "info") {
  const timestamp = new Date().toISOString().split("T")[1].split(".")[0];
  appState.consoleLogs.push({
    message,
    type,
    timestamp,
    time: Date.now(),
  });

  // Keep only last 1000 logs
  if (appState.consoleLogs.length > 1000) {
    appState.consoleLogs.shift();
  }

  renderConsoleLogs();
}

function renderConsoleLogs() {
  const container = document.getElementById("console-output");
  const filter = appState.currentConsoleFilter;

  // Filter logs
  let filteredLogs = appState.consoleLogs;
  if (filter !== "all") {
    const typeMap = {
      info: "info",
      errors: "error",
      warnings: "warning",
    };
    filteredLogs = appState.consoleLogs.filter(
      (log) => log.type === typeMap[filter]
    );
  }

  // Render
  container.innerHTML = filteredLogs
    .map((log) => {
      const typeClass =
        log.type === "error"
          ? "console-line-error"
          : log.type === "warning"
          ? "console-line-warning"
          : log.type === "success"
          ? "console-line-success"
          : "console-line-info";

      return `
            <div class="console-line ${typeClass}">
                <span class="console-timestamp">[${log.timestamp}]</span>
                <span>${escapeHtml(log.message)}</span>
            </div>
        `;
    })
    .join("");

  // Auto-scroll to bottom
  container.scrollTop = container.scrollHeight;
}

function exportLogs() {
  const logsText = appState.consoleLogs
    .map(
      (log) => `[${log.timestamp}] [${log.type.toUpperCase()}] ${log.message}`
    )
    .join("\n");

  const blob = new Blob([logsText], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ml-suite-logs-${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);

  logToConsole("Logs exported successfully", "success");
}

// OCR Module
function initializeOCRModule() {
  const uploadZone = document.getElementById("ocr-upload-zone");
  const fileInput = document.getElementById("ocr-file-input");
  const resultsContainer = document.getElementById("ocr-results");

  // Click to upload
  uploadZone.addEventListener("click", () => fileInput.click());

  // File input change
  fileInput.addEventListener("change", (e) => {
    handleOCRUpload(e.target.files);
  });

  // Drag and drop
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    handleOCRUpload(e.dataTransfer.files);
  });
}

async function handleOCRUpload(files) {
  const formData = new FormData();
  Array.from(files).forEach((file) => {
    formData.append("files", file);
  });

  const resultsContainer = document.getElementById("ocr-results");
  resultsContainer.innerHTML = `
        <div class="ocr-progress">
            <div class="spinner"></div>
            <h3>Processing OCR...</h3>
            <p>Processing ${files.length} file(s). Please wait...</p>
            <div class="progress-bar-container">
                <div class="progress-bar" id="ocr-progress-bar" style="width: 0%"></div>
            </div>
            <p class="progress-text" id="ocr-progress-text">Starting...</p>
        </div>
    `;

  logToConsole(`OCR: Processing ${files.length} file(s)`, "info");

  // Simulate progress for better UX (actual processing happens server-side)
  let progress = 0;
  const progressInterval = setInterval(() => {
    progress += 5;
    if (progress <= 90) {
      document.getElementById("ocr-progress-bar").style.width = `${progress}%`;
      document.getElementById(
        "ocr-progress-text"
      ).textContent = `Processing... ${progress}%`;
    }
  }, 500);

  try {
    const response = await fetch("/api/ocr/upload", {
      method: "POST",
      body: formData,
    });

    clearInterval(progressInterval);
    document.getElementById("ocr-progress-bar").style.width = "100%";
    document.getElementById("ocr-progress-text").textContent = "Complete!";

    const data = await response.json();

    setTimeout(() => {
      displayOCRResults(data.results);
      logToConsole(
        `OCR: Completed processing ${data.results.length} file(s)`,
        "success"
      );
    }, 300);
  } catch (error) {
    clearInterval(progressInterval);
    resultsContainer.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
    logToConsole(`OCR Error: ${error.message}`, "error");
  }
}

function displayOCRResults(results) {
  const container = document.getElementById("ocr-results");
  container.innerHTML = "";

  // Summary header
  const totalWords = results.reduce((sum, r) => sum + (r.word_count || 0), 0);
  const totalChars = results.reduce((sum, r) => sum + (r.char_count || 0), 0);
  const successCount = results.filter((r) => r.success).length;

  container.innerHTML += `
        <div class="ocr-summary">
            <h4>Processing Complete</h4>
            <p>${successCount}/${results.length} files processed | ${totalWords} words | ${totalChars} characters</p>
        </div>
    `;

  results.forEach((result) => {
    const card = document.createElement("div");
    card.className = "result-card";

    if (result.success) {
      const wordCount = result.word_count || 0;
      const charCount = result.char_count || 0;
      card.innerHTML = `
                <h4>${escapeHtml(result.filename)}</h4>
                <div class="ocr-stats">
                    <span>${wordCount} words</span>
                    <span>${charCount} characters</span>
                </div>
                <div class="result-text">${
                  escapeHtml(result.text) || "<em>No text detected</em>"
                }</div>
                <button class="btn btn-secondary copy-btn" onclick="copyToClipboard(\`${escapeHtml(
                  result.text
                ).replace(/`/g, "\\`")}\`)">Copy Text</button>
            `;
    } else {
      card.innerHTML = `<div class="error-message">${escapeHtml(
        result.filename
      )}: ${escapeHtml(result.error)}</div>`;
    }

    container.appendChild(card);
  });
}

// Text Generation Module - Removed

// Model Trainer Module
function initializeTrainerModule() {
  const csvUploadZone = document.getElementById("csv-upload-zone");
  const csvFileInput = document.getElementById("csv-file-input");

  csvUploadZone.addEventListener("click", () => csvFileInput.click());
  csvFileInput.addEventListener("change", (e) =>
    handleCSVUpload(e.target.files[0])
  );

  // Drag and drop
  csvUploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    csvUploadZone.classList.add("dragover");
  });

  csvUploadZone.addEventListener("dragleave", () => {
    csvUploadZone.classList.remove("dragover");
  });

  csvUploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    csvUploadZone.classList.remove("dragover");
    handleCSVUpload(e.dataTransfer.files[0]);
  });

  // API Import
  document
    .getElementById("import-api-btn")
    .addEventListener("click", handleAPIImport);

  // Model category selection
  document
    .getElementById("model-category-select")
    .addEventListener("change", handleCategoryChange);

  // Model type selection
  document
    .getElementById("model-type-select")
    .addEventListener("change", handleModelTypeChange);

  // CSV range refresh
  const refreshPreviewBtn = document.getElementById("refresh-preview-btn");
  if (refreshPreviewBtn) {
    refreshPreviewBtn.addEventListener("click", refreshCSVPreview);
  }

  // Hyperparameter tuning toggle
  document
    .getElementById("enable-hyperparameter-tuning")
    .addEventListener("change", function (e) {
      document.getElementById("hyperparameter-tuning-section").style.display = e
        .target.checked
        ? "block"
        : "none";
    });

  // Custom code toggle
  document
    .getElementById("enable-custom-code")
    .addEventListener("change", function (e) {
      document.getElementById("custom-code-section").style.display = e.target
        .checked
        ? "block"
        : "none";
    });

  document.getElementById("train-btn").addEventListener("click", startTraining);

  // Load initial model categories and populate models
  loadModelCategories();
}

function handleModelTypeChange(e) {
  const modelType = e.target.value;
  const paramsContainer = document.getElementById("model-params-container");
  const paramsSection = document.getElementById("model-parameters");

  // Clear previous parameters
  paramsContainer.innerHTML = "";

  // Handle GPU settings based on model type
  // Deep learning models (neural networks) should use CPU only to avoid threading issues
  const isDeepLearningModel = modelType === "neural_network";
  const gpuEnabledCheckbox = document.getElementById("setting-gpu-enabled");
  const gpuForceCpuCheckbox = document.getElementById("setting-gpu-force-cpu");

  if (isDeepLearningModel) {
    // Disable GPU for neural networks and force CPU mode
    if (gpuEnabledCheckbox) {
      gpuEnabledCheckbox.disabled = true;
      gpuEnabledCheckbox.checked = false;
    }
    if (gpuForceCpuCheckbox) {
      gpuForceCpuCheckbox.disabled = true;
      gpuForceCpuCheckbox.checked = true;
    }
  } else {
    // Re-enable GPU settings for non-deep-learning models
    if (gpuEnabledCheckbox) {
      gpuEnabledCheckbox.disabled = false;
    }
    if (gpuForceCpuCheckbox) {
      gpuForceCpuCheckbox.disabled = false;
    }
  }

  // Show parameters based on model type
  if (modelType === "polynomial") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Polynomial Degree (2-5)</label>
                <input type="number" id="param-degree" class="terminal-input" value="2" min="2" max="5">
                <p class="form-help">Higher degree = more complex curves</p>
            </div>
        `;
  } else if (modelType === "ridge" || modelType === "lasso") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Alpha (Regularization Strength)</label>
                <input type="number" id="param-alpha" class="terminal-input" value="1.0" min="0.01" max="100" step="0.1">
                <p class="form-help">Higher = more regularization (prevents overfitting)</p>
            </div>
        `;
  } else if (modelType === "random_forest") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Number of Trees</label>
                <input type="number" id="param-n_estimators" class="terminal-input" value="100" min="10" max="500">
                <p class="form-help">More trees = better accuracy but slower</p>
            </div>
            <div class="setting-item">
                <label>Max Depth (optional)</label>
                <input type="number" id="param-max_depth" class="terminal-input" placeholder="None" min="1" max="50">
                <p class="form-help">Leave empty for unlimited depth</p>
            </div>
        `;
  } else if (modelType === "gradient_boosting" || modelType === "xgboost") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Number of Estimators</label>
                <input type="number" id="param-n_estimators" class="terminal-input" value="100" min="10" max="500">
            </div>
            <div class="setting-item">
                <label>Learning Rate</label>
                <input type="number" id="param-learning_rate" class="terminal-input" value="0.1" min="0.01" max="1" step="0.01">
                <p class="form-help">Lower = more careful learning</p>
            </div>
            <div class="setting-item">
                <label>Max Depth</label>
                <input type="number" id="param-max_depth" class="terminal-input" value="${
                  modelType === "xgboost" ? 6 : 3
                }" min="1" max="15">
            </div>
        `;
  } else if (modelType === "svr" || modelType === "svc") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Kernel Type</label>
                <select id="param-kernel" class="terminal-select">
                    <option value="rbf">RBF (Radial Basis Function)</option>
                    <option value="linear">Linear</option>
                    <option value="poly">Polynomial</option>
                    <option value="sigmoid">Sigmoid</option>
                </select>
            </div>
            <div class="setting-item">
                <label>C (Regularization Parameter)</label>
                <input type="number" id="param-C" class="terminal-input" value="1.0" min="0.1" max="100" step="0.1">
            </div>
        `;
  } else if (modelType === "logistic_regression") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>C (Inverse Regularization)</label>
                <input type="number" id="param-C" class="terminal-input" value="1.0" min="0.01" max="100" step="0.1">
                <p class="form-help">Higher = less regularization</p>
            </div>
            <div class="setting-item">
                <label>Max Iterations</label>
                <input type="number" id="param-max_iter" class="terminal-input" value="1000" min="100" max="5000" step="100">
            </div>
        `;
  } else if (
    modelType === "random_forest_classifier" ||
    modelType === "gradient_boosting_classifier" ||
    modelType === "xgboost_classifier"
  ) {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Number of Estimators</label>
                <input type="number" id="param-n_estimators" class="terminal-input" value="100" min="10" max="500">
            </div>
            <div class="setting-item">
                <label>Learning Rate</label>
                <input type="number" id="param-learning_rate" class="terminal-input" value="0.1" min="0.01" max="1" step="0.01">
                <p class="form-help">Lower = more careful learning</p>
            </div>
            <div class="setting-item">
                <label>Max Depth</label>
                <input type="number" id="param-max_depth" class="terminal-input" value="${
                  modelType === "xgboost_classifier" ? 6 : 3
                }" min="1" max="15">
            </div>
        `;
  } else if (modelType === "kmeans") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Number of Clusters</label>
                <input type="number" id="param-n_clusters" class="terminal-input" value="3" min="2" max="20">
                <p class="form-help">How many groups to create</p>
            </div>
            <div class="setting-item">
                <label>Max Iterations</label>
                <input type="number" id="param-max_iter" class="terminal-input" value="300" min="100" max="1000" step="100">
            </div>
        `;
  } else if (modelType === "dbscan") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Epsilon (Max Distance)</label>
                <input type="number" id="param-eps" class="terminal-input" value="0.5" min="0.1" max="10" step="0.1">
                <p class="form-help">Maximum distance between samples in a cluster</p>
            </div>
            <div class="setting-item">
                <label>Min Samples</label>
                <input type="number" id="param-min_samples" class="terminal-input" value="5" min="2" max="20">
                <p class="form-help">Minimum samples in a neighborhood</p>
            </div>
        `;
  } else if (modelType === "hierarchical") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Number of Clusters</label>
                <input type="number" id="param-n_clusters" class="terminal-input" value="3" min="2" max="20">
            </div>
            <div class="setting-item">
                <label>Linkage Method</label>
                <select id="param-linkage" class="terminal-select">
                    <option value="ward" selected>Ward</option>
                    <option value="complete">Complete</option>
                    <option value="average">Average</option>
                    <option value="single">Single</option>
                </select>
            </div>
        `;
  } else if (modelType === "pca") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Number of Components</label>
                <input type="number" id="param-n_components" class="terminal-input" value="2" min="1" max="10">
                <p class="form-help">Dimensions to reduce to (2 for visualization)</p>
            </div>
        `;
  } else if (modelType === "neural_network") {
    paramsSection.style.display = "block";
    paramsContainer.innerHTML = `
            <div class="setting-item">
                <label>Hidden Layers (comma-separated)</label>
                <input type="text" id="param-hidden_layers" class="terminal-input" value="128, 64, 32" placeholder="e.g., 256, 128, 64">
                <p class="form-help">Number of neurons in each hidden layer</p>
            </div>
            <div class="setting-item">
                <label>Number of Epochs</label>
                <input type="number" id="param-epochs" class="terminal-input" value="100" min="10" max="1000" step="10">
                <p class="form-help">Training iterations (higher = longer training)</p>
            </div>
            <div class="setting-item">
                <label>Learning Rate</label>
                <input type="number" id="param-learning_rate" class="terminal-input" value="0.001" min="0.0001" max="0.1" step="0.0001">
                <p class="form-help">Optimization step size (lower = more careful learning)</p>
            </div>
            <div class="setting-item">
                <label>Batch Size</label>
                <input type="number" id="param-batch_size" class="terminal-input" value="32" min="8" max="256" step="8">
                <p class="form-help">Number of samples per training update</p>
            </div>
            <div class="setting-item">
                <label>Dropout Rate</label>
                <input type="number" id="param-dropout_rate" class="terminal-input" value="0.2" min="0" max="0.5" step="0.05">
                <p class="form-help">Regularization to prevent overfitting (0-0.5)</p>
            </div>
            <div class="setting-item">
                <label>Weight Decay (L2 Regularization)</label>
                <input type="number" id="param-weight_decay" class="terminal-input" value="0.00001" min="0" max="0.001" step="0.00001">
                <p class="form-help">L2 regularization strength (prevents overfitting)</p>
            </div>
        `;
  } else {
    paramsSection.style.display = "none";
  }
}

function getModelParams(modelType) {
  const params = {};

  if (modelType === "polynomial") {
    params.degree = parseInt(
      document.getElementById("param-degree")?.value || 2
    );
  } else if (modelType === "ridge" || modelType === "lasso") {
    params.alpha = parseFloat(
      document.getElementById("param-alpha")?.value || 1.0
    );
  } else if (
    modelType === "random_forest" ||
    modelType === "random_forest_classifier"
  ) {
    params.n_estimators = parseInt(
      document.getElementById("param-n_estimators")?.value || 100
    );
    const maxDepth = document.getElementById("param-max_depth")?.value;
    params.max_depth = maxDepth ? parseInt(maxDepth) : null;
  } else if (
    modelType === "gradient_boosting" ||
    modelType === "xgboost" ||
    modelType === "gradient_boosting_classifier" ||
    modelType === "xgboost_classifier"
  ) {
    params.n_estimators = parseInt(
      document.getElementById("param-n_estimators")?.value || 100
    );
    params.learning_rate = parseFloat(
      document.getElementById("param-learning_rate")?.value || 0.1
    );
    params.max_depth = parseInt(
      document.getElementById("param-max_depth")?.value || 3
    );
  } else if (modelType === "svr" || modelType === "svc") {
    params.kernel = document.getElementById("param-kernel")?.value || "rbf";
    params.C = parseFloat(document.getElementById("param-C")?.value || 1.0);
  } else if (modelType === "logistic_regression") {
    params.C = parseFloat(document.getElementById("param-C")?.value || 1.0);
    params.max_iter = parseInt(
      document.getElementById("param-max_iter")?.value || 1000
    );
  } else if (modelType === "kmeans") {
    params.n_clusters = parseInt(
      document.getElementById("param-n_clusters")?.value || 3
    );
    params.max_iter = parseInt(
      document.getElementById("param-max_iter")?.value || 300
    );
  } else if (modelType === "dbscan") {
    params.eps = parseFloat(document.getElementById("param-eps")?.value || 0.5);
    params.min_samples = parseInt(
      document.getElementById("param-min_samples")?.value || 5
    );
  } else if (modelType === "hierarchical") {
    params.n_clusters = parseInt(
      document.getElementById("param-n_clusters")?.value || 3
    );
    params.linkage = document.getElementById("param-linkage")?.value || "ward";
  } else if (modelType === "pca") {
    params.n_components = parseInt(
      document.getElementById("param-n_components")?.value || 2
    );
  } else if (modelType === "neural_network") {
    // Parse hidden layers from comma-separated string
    const hiddenLayersStr =
      document.getElementById("param-hidden_layers")?.value || "128, 64, 32";
    params.hidden_layers = hiddenLayersStr
      .split(",")
      .map((x) => parseInt(x.trim()))
      .filter((x) => !isNaN(x));
    params.epochs = parseInt(
      document.getElementById("param-epochs")?.value || 100
    );
    params.learning_rate = parseFloat(
      document.getElementById("param-learning_rate")?.value || 0.001
    );
    params.batch_size = parseInt(
      document.getElementById("param-batch_size")?.value || 32
    );
    params.dropout_rate = parseFloat(
      document.getElementById("param-dropout_rate")?.value || 0.2
    );
    params.weight_decay = parseFloat(
      document.getElementById("param-weight_decay")?.value || 1e-5
    );
  }

  return params;
}

function getTuningTimeEstimate(method) {
  const estimates = {
    random: "2-10 minutes",
    halving: "1-5 minutes",
    grid: "10-60 minutes",
    bayesian: "5-20 minutes",
  };
  return estimates[method] || "5-20 minutes";
}

async function handleAPIImport() {
  const url = document.getElementById("api-url").value;
  const method = document.getElementById("api-method").value;
  const headersText = document.getElementById("api-headers").value;
  const bodyText = document.getElementById("api-body").value;

  if (!url) {
    alert("Please enter an API URL");
    logToConsole("API Import: No URL provided", "warning");
    return;
  }

  const previewContainer = document.getElementById("csv-preview");
  previewContainer.innerHTML =
    '<div class="spinner"></div> Importing from API...';

  logToConsole(`API Import: Fetching data from ${url}`, "info");

  try {
    // Parse headers
    let headers = {};
    if (headersText.trim()) {
      try {
        headers = JSON.parse(headersText);
      } catch (e) {
        previewContainer.innerHTML =
          '<div class="error-message">Invalid JSON in headers</div>';
        logToConsole("API Import: Invalid JSON in headers", "error");
        return;
      }
    }

    // Parse body
    let body = {};
    if (bodyText.trim() && method === "POST") {
      try {
        body = JSON.parse(bodyText);
      } catch (e) {
        previewContainer.innerHTML =
          '<div class="error-message">Invalid JSON in body</div>';
        logToConsole("API Import: Invalid JSON in body", "error");
        return;
      }
    }

    const response = await fetch("/api/data/import-api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, method, headers, body }),
    });

    const data = await response.json();

    if (data.success) {
      appState.currentDataset = data;
      displayCSVPreview(data);
      showDataPreparationStep(data);
      logToConsole(
        `API Import: Successfully imported ${data.row_count} rows`,
        "success"
      );
    } else {
      previewContainer.innerHTML = `<div class="error-message">${data.error}</div>`;
      logToConsole(`API Import Error: ${data.error}`, "error");
    }
  } catch (error) {
    previewContainer.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
    logToConsole(`API Import Error: ${error.message}`, "error");
  }
}

// Global storage for model data
let modelCategoriesData = {};
let availableModelsData = {};

async function loadModelCategories() {
  try {
    // Fetch categories
    const categoriesResponse = await fetch("/api/models/categories");
    const categoriesData = await categoriesResponse.json();
    modelCategoriesData = categoriesData.categories;

    // Fetch available models
    const modelsResponse = await fetch("/api/models/available");
    const modelsData = await modelsResponse.json();
    availableModelsData = modelsData.models;

    // Initialize with default category (supervised_regression)
    handleCategoryChange({ target: { value: "supervised_regression" } });

    logToConsole("Model categories and types loaded successfully", "success");
  } catch (error) {
    logToConsole(`Error loading model categories: ${error.message}`, "error");
  }
}

function handleCategoryChange(e) {
  const category = e.target.value;
  const categoryData = modelCategoriesData[category];

  if (!categoryData) {
    console.error("Category not found:", category);
    return;
  }

  // Update category description
  const descriptionElem = document.getElementById("category-description");
  if (descriptionElem) {
    descriptionElem.textContent = `${categoryData.description} (Use cases: ${categoryData.use_cases})`;
  }

  // Populate model type dropdown
  const modelTypeSelect = document.getElementById("model-type-select");
  modelTypeSelect.innerHTML = "";

  categoryData.models.forEach((modelType, index) => {
    const modelData = availableModelsData[modelType];
    if (modelData) {
      const option = document.createElement("option");
      option.value = modelType;
      option.textContent = modelData.name;
      if (index === 0) option.selected = true;
      modelTypeSelect.appendChild(option);
    }
  });

  // Trigger model type change to load parameters
  const changeEvent = new Event("change");
  modelTypeSelect.dispatchEvent(changeEvent);
}

async function refreshCSVPreview() {
  if (!appState.currentDataset) {
    alert("No dataset loaded. Please upload a CSV first.");
    return;
  }

  const rowStart = parseInt(document.getElementById("row-start").value) || 0;
  const rowEnd = parseInt(document.getElementById("row-end").value) || 10;
  const colStart = parseInt(document.getElementById("col-start").value) || 0;
  const colEnd = parseInt(document.getElementById("col-end").value) || 999;

  logToConsole(
    `Refreshing CSV preview: rows ${rowStart}-${rowEnd}, columns ${colStart}-${colEnd}`,
    "info"
  );
  alert(
    `Preview refresh requested.\nRows: ${rowStart}-${rowEnd}\nColumns: ${colStart}-${colEnd}\n\nNote: Full preview refresh requires re-uploading the file with new range parameters.`
  );
}

async function handleCSVUpload(file) {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const previewContainer = document.getElementById("csv-preview");
  previewContainer.innerHTML = '<div class="spinner"></div> Processing CSV...';

  logToConsole(`Model Trainer: Uploading ${file.name}`, "info");

  try {
    const response = await fetch("/api/data/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      appState.currentDataset = data;
      appState.droppedColumns = new Set(); // Clear dropped columns for new dataset
      displayCSVPreview(data);
      showDataPreparationStep(data);

      // Show CSV range controls
      const rangeControls = document.getElementById("csv-range-controls");
      if (rangeControls) {
        rangeControls.style.display = "block";
        // Set max values based on dataset
        document.getElementById("row-end").max = data.row_count;
        document.getElementById("col-end").max = data.column_count;
      }

      logToConsole(
        `Model Trainer: CSV loaded successfully (${data.row_count} rows, ${data.column_count} columns)`,
        "success"
      );
    } else {
      previewContainer.innerHTML = `<div class="error-message">${data.error}</div>`;
      logToConsole(`Model Trainer Error: ${data.error}`, "error");
    }
  } catch (error) {
    previewContainer.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
    logToConsole(`Model Trainer Error: ${error.message}`, "error");
  }
}

function displayCSVPreview(data) {
  const container = document.getElementById("csv-preview");

  let tableHTML = `
        <div class="success-message">
            File uploaded: ${data.filename} (${data.row_count} rows, ${
    data.column_count
  } columns)
        </div>
        <table class="preview-table">
            <thead><tr>
                ${data.columns
                  .map((col) => `<th>${escapeHtml(col.name)}</th>`)
                  .join("")}
            </tr></thead>
            <tbody>
    `;

  data.preview.forEach((row) => {
    tableHTML += "<tr>";
    data.columns.forEach((col) => {
      const value =
        row[col.name] !== null && row[col.name] !== undefined
          ? row[col.name]
          : "";
      tableHTML += `<td>${escapeHtml(String(value))}</td>`;
    });
    tableHTML += "</tr>";
  });

  tableHTML += "</tbody></table>";
  container.innerHTML = tableHTML;
}

function showDataPreparationStep(data) {
  document.getElementById("step-prepare").style.display = "block";

  // Initialize dropped columns set if not exists
  if (!appState.droppedColumns) {
    appState.droppedColumns = new Set();
  }

  // Column selector
  const columnSelector = document.getElementById("column-selector");

  // Show columns with high missing values
  const highMissingCols = data.columns.filter((col) => {
    const missingPercent = (col.null_count / data.row_count) * 100;
    return missingPercent > 30; // More than 30% missing
  });

  let headerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h4 style="margin: 0;">Select Columns</h4>
            <div>
                <button class="btn btn-secondary" id="select-all-features" style="padding: 8px 16px; margin-right: 10px;">Select All Features</button>
                <button class="btn btn-secondary" id="deselect-all-features" style="padding: 8px 16px;">Deselect All</button>
            </div>
        </div>
    `;

  // Add warning for columns with high missing values
  if (highMissingCols.length > 0) {
    headerHTML += `
            <div class="warning-message" style="margin-bottom: 15px;">
                <strong>⚠️ Columns with >30% missing values:</strong><br>
                ${highMissingCols
                  .map((col) => {
                    const percent = (
                      (col.null_count / data.row_count) *
                      100
                    ).toFixed(1);
                    return `${col.name} (${percent}% missing)`;
                  })
                  .join(", ")}<br>
                <small>Consider dropping these columns to improve model quality.</small>
            </div>
        `;
  }

  columnSelector.innerHTML = headerHTML;

  data.columns.forEach((col) => {
    const missingPercent = (col.null_count / data.row_count) * 100;
    const isDropped = appState.droppedColumns.has(col.name);

    if (isDropped) return; // Skip dropped columns

    const item = document.createElement("div");
    item.className = "column-item";
    item.id = `column-item-${col.name}`;

    let columnHTML = `
            <input type="checkbox" id="col-${col.name}" data-column="${
      col.name
    }" class="feature-checkbox">
            <label for="col-${col.name}">${escapeHtml(col.name)}</label>
            <span class="column-type">${col.type}</span>
        `;

    // Show missing value warning
    if (col.null_count > 0) {
      columnHTML += `<span class="column-warning" style="color: var(--warning); font-size: 0.85em; margin-left: 10px;">(${
        col.null_count
      } missing - ${missingPercent.toFixed(1)}%)</span>`;
    }

    columnHTML += `
            <input type="radio" name="target-column" id="target-${col.name}" value="${col.name}" class="target-radio">
            <label for="target-${col.name}" style="margin-left: 20px;">Target</label>
        `;

    // Add drop button for columns with high missing values
    if (missingPercent > 30) {
      columnHTML += `
                <button class="btn btn-danger" onclick="dropColumn('${col.name}')" style="margin-left: 15px; padding: 4px 12px; font-size: 0.85em;">
                    Drop Column
                </button>
            `;
    }

    item.innerHTML = columnHTML;
    columnSelector.appendChild(item);
  });

  // Add event listeners for select/deselect all
  document
    .getElementById("select-all-features")
    .addEventListener("click", () => {
      document
        .querySelectorAll(".feature-checkbox")
        .forEach((cb) => (cb.checked = true));
      logToConsole("Model Trainer: Selected all feature columns", "info");
    });

  document
    .getElementById("deselect-all-features")
    .addEventListener("click", () => {
      document
        .querySelectorAll(".feature-checkbox")
        .forEach((cb) => (cb.checked = false));
      logToConsole("Model Trainer: Deselected all feature columns", "info");
    });

  // Data cleaning
  const cleaningContainer = document.getElementById("data-cleaning");
  cleaningContainer.innerHTML = "<h4>Handle Missing Values</h4>";

  data.columns.forEach((col) => {
    if (col.null_count > 0) {
      const item = document.createElement("div");
      item.className = "cleaning-item";
      item.innerHTML = `
                <span>${escapeHtml(col.name)} (${
        col.null_count
      } missing):</span>
                <select id="clean-${col.name}" class="terminal-select">
                    <option value="drop">Drop rows</option>
                    <option value="mean">Fill with mean</option>
                    <option value="median">Fill with median</option>
                    <option value="mode">Fill with mode</option>
                    <option value="zero">Fill with zero</option>
                </select>
            `;
      cleaningContainer.appendChild(item);
    }
  });

  // Train-test split
  const splitContainer = document.getElementById("train-test-split");
  splitContainer.innerHTML = `
        <h4>Train-Test Split</h4>
        <input type="range" id="split-slider" class="split-slider" min="50" max="95" value="80">
        <div class="split-display">
            <span id="split-display-text">Training: ${Math.round(
              data.row_count * 0.8
            )} rows, Testing: ${Math.round(data.row_count * 0.2)} rows</span>
        </div>
    `;

  document.getElementById("split-slider").addEventListener("input", (e) => {
    const trainSize = e.target.value / 100;
    const trainRows = Math.round(data.row_count * trainSize);
    const testRows = data.row_count - trainRows;
    document.getElementById(
      "split-display-text"
    ).textContent = `Training: ${trainRows} rows, Testing: ${testRows} rows`;
  });

  // Show training step
  document.getElementById("step-train").style.display = "block";
}

async function startTraining() {
  const dataset = appState.currentDataset;
  if (!dataset) {
    alert("Please upload a dataset first");
    logToConsole("Training Error: No dataset loaded", "error");
    return;
  }

  // Collect configuration
  const featureColumns = Array.from(
    document.querySelectorAll(".feature-checkbox:checked")
  )
    .map((cb) => cb.getAttribute("data-column"))
    .filter(
      (col) => !appState.droppedColumns || !appState.droppedColumns.has(col)
    ); // Exclude dropped columns

  const targetColumn = document.querySelector(
    'input[name="target-column"]:checked'
  )?.value;

  // Ensure target column is not dropped
  if (
    targetColumn &&
    appState.droppedColumns &&
    appState.droppedColumns.has(targetColumn)
  ) {
    alert(
      "Target column has been dropped. Please select a different target column."
    );
    logToConsole("Training Error: Target column is dropped", "error");
    return;
  }

  if (!targetColumn) {
    alert("Please select a target column");
    logToConsole("Training Error: No target column selected", "error");
    return;
  }

  if (featureColumns.length === 0) {
    alert("Please select at least one feature column");
    logToConsole("Training Error: No feature columns selected", "error");
    return;
  }

  const missingValueStrategy = {};
  dataset.columns.forEach((col) => {
    const select = document.getElementById(`clean-${col.name}`);
    if (select) {
      missingValueStrategy[col.name] = select.value;
    }
  });

  const trainSize = document.getElementById("split-slider").value / 100;
  const modelName =
    document.getElementById("model-name").value || `model_${Date.now()}`;

  // Get model type and parameters
  const modelType = document.getElementById("model-type-select").value;
  const modelParams = getModelParams(modelType);

  // Get hyperparameter tuning settings
  const enableTuning = document.getElementById(
    "enable-hyperparameter-tuning"
  ).checked;
  const tuningConfig = enableTuning
    ? {
        method: document.getElementById("tuning-method").value,
        cv_folds: parseInt(document.getElementById("tuning-cv-folds").value),
        n_iter: parseInt(document.getElementById("tuning-iterations").value),
      }
    : null;

  // Get custom code if enabled
  const enableCustomCode =
    document.getElementById("enable-custom-code").checked;
  const customCode = enableCustomCode
    ? document.getElementById("custom-code").value
    : null;

  const progressContainer = document.getElementById("training-progress");
  const resultsContainer = document.getElementById("training-results");

  const trainingMessage = `
        <div style="background: var(--bg-surface); border: 2px solid var(--accent); border-radius: 12px; padding: 20px; text-align: center;">
            <div class="spinner" style="margin: 0 auto 16px;"></div>
            <h4 style="color: var(--accent); margin-bottom: 8px;">Training in Progress...</h4>
            <p style="color: var(--text-secondary); margin-bottom: 4px;">${modelType
              .replace(/_/g, " ")
              .toUpperCase()}</p>
            ${
              enableTuning
                ? `<p style="color: var(--warning); font-size: 0.9em;">⚠️ Hyperparameter tuning enabled (${
                    tuningConfig.method
                  })</p>
            <p style="color: var(--text-secondary); font-size: 0.85em;">Estimated time: ${getTuningTimeEstimate(
              tuningConfig.method
            )}</p>`
                : ""
            }
            <div style="margin-top: 12px; font-size: 0.85em; color: var(--text-secondary);">
                <span>Features: ${featureColumns.length}</span> | 
                <span>Samples: ${dataset.row_count}</span>
            </div>
        </div>
    `;

  progressContainer.innerHTML = trainingMessage;
  resultsContainer.innerHTML = "";

  logToConsole(`Training: Starting ${modelType} for ${modelName}`, "info");
  if (enableTuning) {
    logToConsole(
      `Training: Hyperparameter tuning enabled (${tuningConfig.method})`,
      "info"
    );
  }
  logToConsole(
    `Training: Target=${targetColumn}, Features=[${featureColumns.join(", ")}]`,
    "info"
  );

  // Force CPU-only mode for neural networks to avoid PyTorch threading issues
  const isDeepLearningModel = modelType === "neural_network";
  if (isDeepLearningModel) {
    logToConsole(
      `Training: Deep Learning model detected - forcing CPU-only mode for stability`,
      "warning"
    );
  }

  try {
    const response = await fetch("/api/model/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: dataset.dataset_id,
        target_column: targetColumn,
        feature_columns: featureColumns,
        missing_value_strategy: missingValueStrategy,
        remove_duplicates: false,
        train_size: trainSize,
        model_name: modelName,
        model_type: modelType,
        model_params: modelParams,
        hyperparameter_tuning: tuningConfig,
        custom_code: customCode,
        chart_settings: appState.settings.showCharts || {},
        // Force CPU-only mode for deep learning models to avoid PyTorch threading issues
        gpu_enabled: isDeepLearningModel ? false : appState.settings.gpuEnabled,
        gpu_force_cpu: isDeepLearningModel
          ? true
          : appState.settings.gpuForceCpu,
      }),
    });

    const data = await response.json();

    if (data.success) {
      progressContainer.innerHTML =
        '<div class="success-message">Training completed!</div>';
      displayTrainingResults(data);

      // Log hardware info
      let hardwareInfo = "CPU";
      if (data.hardware_info) {
        if (data.hardware_info.gpu_accelerated) {
          const backend =
            data.hardware_info.gpu_backend || data.hardware_info.device_used;
          hardwareInfo = `GPU (${backend.toUpperCase()})`;
          if (data.hardware_info.gpu_memory_used) {
            hardwareInfo += ` | Memory: ${data.hardware_info.gpu_memory_used}GB`;
          }
        }
      }

      const trainingTime = data.training_time ? `${data.training_time}s` : "";
      logToConsole(
        `Training Hardware: ${hardwareInfo} ${
          trainingTime ? `| Time: ${trainingTime}` : ""
        }`,
        "info"
      );

      // Log success with appropriate metric
      let metricInfo = "";
      if (data.metrics.r2_score !== undefined) {
        metricInfo = `R²=${data.metrics.r2_score.toFixed(4)}`;
      } else if (data.metrics.accuracy !== undefined) {
        metricInfo = `Accuracy=${data.metrics.accuracy.toFixed(4)}`;
      } else if (data.metrics.silhouette_score !== undefined) {
        metricInfo = `Silhouette=${data.metrics.silhouette_score.toFixed(4)}`;
      } else if (data.metrics.n_components !== undefined) {
        metricInfo = `Components=${data.metrics.n_components}`;
      }
      logToConsole(
        `Training: Model "${modelName}" trained successfully ${
          metricInfo ? `(${metricInfo})` : ""
        }`,
        "success"
      );

      // Refresh models list
      if (appState.currentModule === "models") {
        loadModels();
      }
    } else {
      progressContainer.innerHTML = `<div class="error-message">${data.error}</div>`;
      logToConsole(`Training Error: ${data.error}`, "error");
    }
  } catch (error) {
    progressContainer.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
    logToConsole(`Training Error: ${error.message}`, "error");
  }
}

function displayTrainingResults(data) {
  const container = document.getElementById("training-results");
  const settings = appState.settings.showMetrics || {};

  let metricsHTML = "";

  // Display metrics based on model category
  if (data.metrics.r2_score !== undefined) {
    // Regression metrics
    if (settings.r2Score !== false) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">R² Score:</span>
                    <span class="metric-value">${data.metrics.r2_score.toFixed(
                      4
                    )}</span>
                </div>`;
    }

    if (settings.rmse !== false) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">RMSE:</span>
                    <span class="metric-value">${data.metrics.rmse.toFixed(
                      4
                    )}</span>
                </div>`;
    }

    if (settings.mae !== false) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">MAE:</span>
                    <span class="metric-value">${data.metrics.mae.toFixed(
                      4
                    )}</span>
                </div>`;
    }

    if (settings.mse && data.metrics.mse) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">MSE:</span>
                    <span class="metric-value">${data.metrics.mse.toFixed(
                      4
                    )}</span>
                </div>`;
    }

    if (settings.adjustedR2 && data.metrics.adjusted_r2) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">Adjusted R²:</span>
                    <span class="metric-value">${data.metrics.adjusted_r2.toFixed(
                      4
                    )}</span>
                </div>`;
    }
  } else if (data.metrics.accuracy !== undefined) {
    // Classification metrics
    metricsHTML += `
            <div class="metric">
                <span class="metric-label">Accuracy:</span>
                <span class="metric-value">${data.metrics.accuracy.toFixed(
                  4
                )}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Precision:</span>
                <span class="metric-value">${data.metrics.precision.toFixed(
                  4
                )}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Recall:</span>
                <span class="metric-value">${data.metrics.recall.toFixed(
                  4
                )}</span>
            </div>
            <div class="metric">
                <span class="metric-label">F1 Score:</span>
                <span class="metric-value">${data.metrics.f1_score.toFixed(
                  4
                )}</span>
            </div>`;

    if (data.metrics.roc_auc) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">ROC-AUC:</span>
                    <span class="metric-value">${data.metrics.roc_auc.toFixed(
                      4
                    )}</span>
                </div>`;
    }
  } else if (data.metrics.n_clusters !== undefined) {
    // Clustering metrics
    metricsHTML += `
            <div class="metric">
                <span class="metric-label">Number of Clusters:</span>
                <span class="metric-value">${data.metrics.n_clusters}</span>
            </div>`;

    if (data.metrics.silhouette_score !== null) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">Silhouette Score:</span>
                    <span class="metric-value">${data.metrics.silhouette_score.toFixed(
                      4
                    )}</span>
                </div>`;
    }

    if (data.metrics.davies_bouldin_score !== null) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">Davies-Bouldin Score:</span>
                    <span class="metric-value">${data.metrics.davies_bouldin_score.toFixed(
                      4
                    )}</span>
                </div>`;
    }
  } else if (data.metrics.n_components !== undefined) {
    // Dimensionality reduction metrics
    metricsHTML += `
            <div class="metric">
                <span class="metric-label">Components:</span>
                <span class="metric-value">${data.metrics.n_components}</span>
            </div>`;

    if (data.metrics.explained_variance_ratio) {
      metricsHTML += `
                <div class="metric">
                    <span class="metric-label">Total Variance Explained:</span>
                    <span class="metric-value">${(
                      data.metrics.explained_variance_ratio.reduce(
                        (a, b) => a + b,
                        0
                      ) * 100
                    ).toFixed(2)}%</span>
                </div>`;
    }
  }

  let chartsHTML = "";
  if (data.charts) {
    chartsHTML = '<div class="charts-section"><h4>Visualizations</h4>';

    // Regression charts
    if (data.charts.actualVsPredicted) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.actualVsPredicted}" alt="Actual vs Predicted" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    if (data.charts.residualPlot) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.residualPlot}" alt="Residual Plot" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    if (data.charts.distribution) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.distribution}" alt="Distribution Comparison" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    // Classification charts
    if (data.charts.confusionMatrix) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.confusionMatrix}" alt="Confusion Matrix" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    if (data.charts.rocCurve) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.rocCurve}" alt="ROC Curve" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    // Clustering charts
    if (data.charts.clusterScatter) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.clusterScatter}" alt="Cluster Visualization" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    // Dimensionality reduction charts
    if (data.charts.screePlot) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.screePlot}" alt="Scree Plot" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    if (data.charts.transformedScatter) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.transformedScatter}" alt="Transformed Data" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    // Common charts
    if (data.charts.featureImportance) {
      chartsHTML += `
                <div class="chart-container">
                    <img src="${data.charts.featureImportance}" alt="Feature Importance" style="max-width: 100%; border: 1px solid var(--border); border-radius: 4px;">
                </div>`;
    }

    chartsHTML += "</div>";
  }

  // Training time display
  let trainingTimeHTML = "";
  if (data.training_time) {
    trainingTimeHTML = `
            <div class="training-info" style="margin-top: 20px; padding: 12px; background: var(--bg-surface); border: 1px solid var(--success); border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">✓</span>
                    <div>
                        <strong style="color: var(--success);">Training Completed Successfully</strong>
                        <div style="margin-top: 4px; color: var(--text-secondary); font-size: 0.9em;">
                            <span>Time taken: <strong>${
                              data.training_time
                            }s</strong></span>
                            ${
                              data.train_size
                                ? ` | Training samples: <strong>${data.train_size}</strong>`
                                : ""
                            }
                            ${
                              data.test_size
                                ? ` | Test samples: <strong>${data.test_size}</strong>`
                                : ""
                            }
                        </div>
                    </div>
                </div>
            </div>
        `;
  }

  // Coefficients section (only for models that have them)
  let coefficientsHTML = "";
  if (data.coefficients && Object.keys(data.coefficients).length > 0) {
    coefficientsHTML = `
            <div style="margin-top: 20px;">
                <h5>Feature Importance / Coefficients:</h5>
                <div class="result-text">
                    ${Object.entries(data.coefficients)
                      .map(
                        ([k, v]) =>
                          `${k}: ${typeof v === "number" ? v.toFixed(4) : v}`
                      )
                      .join("\n")}
                </div>
                ${
                  data.intercept !== undefined &&
                  data.intercept !== null &&
                  typeof data.intercept === "number"
                    ? `<p style="margin-top: 10px;">Intercept: ${data.intercept.toFixed(
                        4
                      )}</p>`
                    : ""
                }
            </div>
        `;
  }

  container.innerHTML = `
        <div class="result-card">
            <h4>Model: ${escapeHtml(data.model_name)}</h4>
            ${trainingTimeHTML}
            <div class="model-metrics">
                ${metricsHTML}
            </div>
            ${coefficientsHTML}
            ${chartsHTML}
        </div>
    `;
}

// Drop Column Function
function dropColumn(columnName) {
  if (
    !confirm(
      `Are you sure you want to drop column "${columnName}"?\n\nThis column will be permanently excluded from analysis and training.`
    )
  ) {
    return;
  }

  // Add to dropped columns set
  appState.droppedColumns.add(columnName);

  // Remove the column item from UI
  const columnItem = document.getElementById(`column-item-${columnName}`);
  if (columnItem) {
    columnItem.remove();
  }

  // Update dataset columns
  if (appState.currentDataset) {
    appState.currentDataset.columns = appState.currentDataset.columns.filter(
      (col) => col.name !== columnName
    );
  }

  logToConsole(`Column dropped: ${columnName}`, "info");
  alert(
    `Column "${columnName}" has been dropped and will not be used in training.`
  );
}

// Model Management Module
function initializeModelsModule() {
  document
    .getElementById("refresh-models-btn")
    .addEventListener("click", loadModels);
  document
    .getElementById("load-model-btn")
    .addEventListener("click", showModelLoader);
  document
    .getElementById("confirm-load-model-btn")
    .addEventListener("click", loadSelectedModel);
  document
    .getElementById("cancel-load-model-btn")
    .addEventListener("click", () => {
      document.getElementById("model-loader").style.display = "none";
    });

  // Bulk selection handlers
  document
    .getElementById("select-all-models-btn")
    .addEventListener("click", toggleSelectAll);
  document
    .getElementById("delete-selected-models-btn")
    .addEventListener("click", deleteSelectedModels);

  loadModels();
}

// Global state for selected models
let selectedModels = new Set();

function toggleSelectAll() {
  const checkboxes = document.querySelectorAll(".model-checkbox");
  const allChecked = Array.from(checkboxes).every((cb) => cb.checked);

  checkboxes.forEach((checkbox) => {
    checkbox.checked = !allChecked;
    if (!allChecked) {
      selectedModels.add(checkbox.dataset.modelId);
    } else {
      selectedModels.delete(checkbox.dataset.modelId);
    }
  });

  updateBulkDeleteButton();
}

function updateBulkDeleteButton() {
  const deleteBtn = document.getElementById("delete-selected-models-btn");
  const countSpan = document.getElementById("selected-count");

  if (selectedModels.size > 0) {
    deleteBtn.style.display = "inline-block";
    countSpan.textContent = selectedModels.size;
  } else {
    deleteBtn.style.display = "none";
  }
}

async function deleteSelectedModels() {
  if (selectedModels.size === 0) return;

  const confirmed = confirm(
    `Are you sure you want to delete ${selectedModels.size} model(s)? This action cannot be undone.`
  );
  if (!confirmed) return;

  const deleteBtn = document.getElementById("delete-selected-models-btn");
  deleteBtn.disabled = true;
  deleteBtn.textContent = "Deleting...";

  let successCount = 0;
  let failCount = 0;

  for (const modelId of selectedModels) {
    try {
      const response = await fetch(`/api/models/${modelId}`, {
        method: "DELETE",
      });

      if (response.ok) {
        successCount++;
      } else {
        failCount++;
      }
    } catch (error) {
      failCount++;
    }
  }

  // Clear selection
  selectedModels.clear();

  // Reset button
  deleteBtn.disabled = false;
  deleteBtn.textContent = `Delete Selected (0)`;
  updateBulkDeleteButton();

  // Show result
  if (failCount === 0) {
    logToConsole(`Successfully deleted ${successCount} model(s)`, "success");
  } else {
    logToConsole(
      `Deleted ${successCount} model(s), failed to delete ${failCount}`,
      "warning"
    );
  }

  // Reload models list
  loadModels();
}

function showModelLoader() {
  const loaderSection = document.getElementById("model-loader");
  loaderSection.style.display = "block";

  // Populate dropdown with available models
  fetch("/api/models")
    .then((res) => res.json())
    .then((data) => {
      const dropdown = document.getElementById("model-select-dropdown");
      dropdown.innerHTML = '<option value="">-- Select a model --</option>';
      data.models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.model_id;
        option.textContent = `${model.model_name} (${
          model.model_type
        }) - R²: ${model.metrics.r2_score?.toFixed(4)}`;
        dropdown.appendChild(option);
      });
    })
    .catch((err) => {
      alert("Error loading models: " + err.message);
    });
}

async function loadSelectedModel() {
  const modelId = document.getElementById("model-select-dropdown").value;
  if (!modelId) {
    alert("Please select a model");
    return;
  }

  try {
    const response = await fetch(`/api/models/${modelId}`);
    const modelData = await response.json();

    // Switch to Model Trainer module
    switchModule("trainer");

    // Store loaded model info for reference
    appState.loadedModel = modelData;

    // Populate the trainer UI with loaded model data
    populateTrainerWithLoadedModel(modelData);

    // Show message
    logToConsole(
      `Loaded model: ${modelData.model_name} (${modelData.model_type})`,
      "success"
    );

    // Show success message with loaded model info
    const message =
      `Model "${modelData.model_name}" loaded successfully!\n\n` +
      `✓ Model type set to: ${modelData.model_type}\n` +
      `✓ Target variable: ${modelData.target}\n` +
      `✓ Feature columns: ${modelData.features.join(", ")}\n\n` +
      `📝 Note: You can adjust model parameters and settings as needed.\n` +
      `The form is now ready for training!`;

    alert(message);

    // Hide loader
    document.getElementById("model-loader").style.display = "none";
  } catch (error) {
    alert("Error loading model: " + error.message);
  }
}

function populateTrainerWithLoadedModel(modelData) {
  // Set model type in dropdown
  const modelTypeSelect = document.getElementById("model-type-select");
  if (modelTypeSelect) {
    modelTypeSelect.value = modelData.model_type;

    // Trigger the change event to show parameters
    const changeEvent = new Event("change");
    modelTypeSelect.dispatchEvent(changeEvent);
  }

  // Set model name with "loaded" indicator
  const modelNameInput = document.getElementById("model-name");
  if (modelNameInput) {
    modelNameInput.value = `${modelData.model_name}_retrained_${Date.now()}`;
  }

  // Note: Model parameters are not stored in metadata, so user will need to set them manually
  // Only the model type is set, and the user can configure parameters as needed

  // Show a visual indicator that a model was loaded
  const stepTrain = document.getElementById("step-train");
  if (stepTrain) {
    // Add a small note at the top of the training step
    const existingNote = stepTrain.querySelector(".loaded-model-note");
    if (existingNote) {
      existingNote.remove();
    }

    const note = document.createElement("div");
    note.className = "loaded-model-note";
    note.style.cssText = `
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid var(--success);
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 14px;
            color: var(--success);
        `;
    note.innerHTML = `
            <strong>📋 Model Configuration Loaded:</strong> ${modelData.model_name}<br>
            <strong>Type:</strong> ${modelData.model_type} |
            <strong>Target:</strong> ${modelData.target} |
            <strong>Features:</strong> ${modelData.features.length} columns<br>
            <small style="color: var(--text-secondary);">You can adjust parameters and train a new model with this setup.</small>
        `;

    const stepHeader = stepTrain.querySelector("h3");
    if (stepHeader) {
      stepHeader.insertAdjacentElement("afterend", note);
    }
  }
}

// Note: populateModelParameters function removed - parameters not stored in model metadata

async function loadModels() {
  const container = document.getElementById("models-list");
  container.innerHTML = '<div class="spinner"></div> Loading models...';

  try {
    const response = await fetch("/api/models");
    const data = await response.json();

    if (data.models.length === 0) {
      container.innerHTML =
        '<div class="result-card">No models found. Train a model in the Model Trainer module.</div>';
      return;
    }

    container.innerHTML = "";
    selectedModels.clear(); // Clear previous selection
    updateBulkDeleteButton();

    data.models.forEach((model) => {
      const card = document.createElement("div");
      card.className = "model-card";
      card.innerHTML = `
                <div class="model-card-header">
                    <div style="display: flex; align-items: flex-start; gap: 12px;">
                        <input type="checkbox" class="model-checkbox" data-model-id="${
                          model.model_id
                        }" style="margin-top: 4px; cursor: pointer; width: 18px; height: 18px;">
                        <div style="flex: 1;">
                            <div class="model-card-title">${escapeHtml(
                              model.model_name
                            )}</div>
                            <div class="model-card-meta">
                                Type: ${model.model_type} | Created: ${new Date(
        model.created_at
      ).toLocaleString()}
                            </div>
                            <div class="model-card-meta">
                                Dataset: ${escapeHtml(model.dataset_name)}
                            </div>
                            ${
                              model.training_time
                                ? `
                                <div class="model-card-meta">
                                    ⏱️ Training: ${model.training_time}s ${
                                    model.hardware_used &&
                                    model.hardware_used.gpu_accelerated
                                      ? `| 🚀 GPU (${model.hardware_used.backend.toUpperCase()})`
                                      : "| 💻 CPU"
                                  }
                                </div>
                            `
                                : ""
                            }
                        </div>
                    </div>
                </div>
                <div class="model-metrics">
                    ${
                      model.metrics.r2_score !== undefined
                        ? `
                        <div class="metric">
                            <span class="metric-label">R² Score:</span>
                            <span class="metric-value">${model.metrics.r2_score.toFixed(
                              4
                            )}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">RMSE:</span>
                            <span class="metric-value">${model.metrics.rmse.toFixed(
                              4
                            )}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">MAE:</span>
                            <span class="metric-value">${model.metrics.mae.toFixed(
                              4
                            )}</span>
                        </div>
                    `
                        : ""
                    }
                    ${
                      model.metrics.accuracy !== undefined
                        ? `
                        <div class="metric">
                            <span class="metric-label">Accuracy:</span>
                            <span class="metric-value">${model.metrics.accuracy.toFixed(
                              4
                            )}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Precision:</span>
                            <span class="metric-value">${model.metrics.precision.toFixed(
                              4
                            )}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">F1 Score:</span>
                            <span class="metric-value">${model.metrics.f1_score.toFixed(
                              4
                            )}</span>
                        </div>
                        ${
                          model.metrics.roc_auc
                            ? `
                            <div class="metric">
                                <span class="metric-label">ROC-AUC:</span>
                                <span class="metric-value">${model.metrics.roc_auc.toFixed(
                                  4
                                )}</span>
                            </div>
                        `
                            : ""
                        }
                    `
                        : ""
                    }
                    ${
                      model.metrics.n_clusters !== undefined
                        ? `
                        <div class="metric">
                            <span class="metric-label">Clusters:</span>
                            <span class="metric-value">${
                              model.metrics.n_clusters
                            }</span>
                        </div>
                        ${
                          model.metrics.silhouette_score
                            ? `
                            <div class="metric">
                                <span class="metric-label">Silhouette Score:</span>
                                <span class="metric-value">${model.metrics.silhouette_score.toFixed(
                                  4
                                )}</span>
                            </div>
                        `
                            : ""
                        }
                    `
                        : ""
                    }
                    ${
                      model.metrics.n_components !== undefined
                        ? `
                        <div class="metric">
                            <span class="metric-label">Components:</span>
                            <span class="metric-value">${
                              model.metrics.n_components
                            }</span>
                        </div>
                        ${
                          model.metrics.explained_variance_ratio
                            ? `
                            <div class="metric">
                                <span class="metric-label">Variance Explained:</span>
                                <span class="metric-value">${(
                                  model.metrics.explained_variance_ratio.reduce(
                                    (a, b) => a + b,
                                    0
                                  ) * 100
                                ).toFixed(2)}%</span>
                            </div>
                        `
                            : ""
                        }
                    `
                        : ""
                    }
                </div>
                <div class="model-actions">
                    <button class="btn btn-secondary" onclick="viewModelDetails('${
                      model.model_id
                    }')">View Details</button>
                    <button class="btn btn-secondary" onclick="downloadModel('${
                      model.model_id
                    }', '${model.model_name}')">Download Model</button>
                    <button class="btn btn-primary" onclick="exportNotebook('${
                      model.model_id
                    }')">Export Notebook</button>
                    <button class="btn btn-danger" onclick="deleteModel('${
                      model.model_id
                    }')">Delete</button>
                </div>
            `;
      container.appendChild(card);

      // Add checkbox event listener
      const checkbox = card.querySelector(".model-checkbox");
      checkbox.addEventListener("change", (e) => {
        if (e.target.checked) {
          selectedModels.add(model.model_id);
        } else {
          selectedModels.delete(model.model_id);
        }
        updateBulkDeleteButton();
      });
    });
  } catch (error) {
    container.innerHTML = `<div class="error-message">Error loading models: ${error.message}</div>`;
  }
}

async function viewModelDetails(modelId) {
  try {
    const response = await fetch(`/api/models/${modelId}`);
    const data = await response.json();

    // Build details string based on available data
    let details = `Model Details:\n\n`;
    details += `Name: ${data.model_name}\n`;
    details += `Type: ${data.model_type}\n`;
    details += `Created: ${new Date(data.created_at).toLocaleString()}\n`;

    // Training info
    if (data.training_time) {
      details += `\nTraining Time: ${data.training_time}s\n`;
      if (data.hardware_used && data.hardware_used.gpu_accelerated) {
        details += `Hardware: GPU (${data.hardware_used.backend.toUpperCase()})\n`;
      } else {
        details += `Hardware: CPU\n`;
      }
    }

    // Dataset info
    details += `\nDataset: ${data.dataset_name}\n`;
    details += `Features: ${data.features.join(", ")}\n`;
    details += `Target: ${data.target}\n`;

    // Metrics
    details += `\nMetrics:\n`;
    if (data.metrics.r2_score !== undefined) {
      details += `  R²: ${data.metrics.r2_score.toFixed(4)}\n`;
      if (data.metrics.rmse)
        details += `  RMSE: ${data.metrics.rmse.toFixed(4)}\n`;
      if (data.metrics.mae)
        details += `  MAE: ${data.metrics.mae.toFixed(4)}\n`;
    } else if (data.metrics.accuracy !== undefined) {
      details += `  Accuracy: ${data.metrics.accuracy.toFixed(4)}\n`;
      if (data.metrics.precision)
        details += `  Precision: ${data.metrics.precision.toFixed(4)}\n`;
      if (data.metrics.recall)
        details += `  Recall: ${data.metrics.recall.toFixed(4)}\n`;
      if (data.metrics.f1_score)
        details += `  F1 Score: ${data.metrics.f1_score.toFixed(4)}\n`;
    } else if (data.metrics.n_clusters !== undefined) {
      details += `  Clusters: ${data.metrics.n_clusters}\n`;
      if (data.metrics.silhouette_score)
        details += `  Silhouette: ${data.metrics.silhouette_score.toFixed(
          4
        )}\n`;
    }

    alert(details);
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}

async function downloadModel(modelId, modelName) {
  try {
    logToConsole(`Downloading model: ${modelId}`, "info");

    // Create a temporary link element
    const link = document.createElement("a");
    link.href = `/api/models/${modelId}/download`;
    link.download = `${modelId}.pkl`;

    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    logToConsole(`Model download started: ${modelName}`, "success");
    alert(
      `Downloading model: ${modelName}\n\nFile: ${modelId}.pkl\n\nCheck your downloads folder!`
    );
  } catch (error) {
    logToConsole(`Model download error: ${error.message}`, "error");
    alert(`Error downloading model: ${error.message}`);
  }
}

async function deleteModel(modelId) {
  if (!confirm("Are you sure you want to delete this model?")) {
    return;
  }

  try {
    const response = await fetch(`/api/models/${modelId}`, {
      method: "DELETE",
    });

    const data = await response.json();

    if (data.success) {
      loadModels();
      logToConsole(`Model deleted: ${modelId}`, "info");
    } else {
      alert(`Error: ${data.error}`);
    }
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}

// Notebook export function moved to notebook_export.js for better organization
// This function is now comprehensive with full visualization code for all model types

/* OLD FUNCTION - REPLACED
async function exportNotebook(modelId) {
    try {
        logToConsole(`Exporting comprehensive Jupyter notebook for model: ${modelId}`, 'info');
        
        // Fetch model details
        const response = await fetch(`/api/models/${modelId}`);
        const model = await response.json();
        
        // Determine model category
        const modelCategory = model.model_category || 'regression';
        
        // Create Jupyter notebook JSON
        const notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        `# ML Model: ${model.model_name}\n`,
                        `\n`,
                        `**Model Type:** ${model.model_type}\n`,
                        `**Created:** ${model.created_at}\n`,
                        `**Dataset:** ${model.dataset_name}\n`,
                        `\n`,
                        `## Model Performance\n`,
                        `- **R² Score:** ${model.metrics.r2_score}\n`,
                        `- **RMSE:** ${model.metrics.rmse}\n`,
                        `- **MAE:** ${model.metrics.mae}\n`,
                        `- **MSE:** ${model.metrics.mse || 'N/A'}\n`,
                        `- **Adjusted R²:** ${model.metrics.adjusted_r2 || 'N/A'}\n`
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "source": [
                        "# Import required libraries\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import joblib\n",
                        "from sklearn.model_selection import train_test_split\n",
                        "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        `## Load Model\n`,
                        `\n`,
                        `Download the model file and update the path below.`
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "source": [
                        `# Load the trained model\n`,
                        `model = joblib.load('${model.model_id}.pkl')\n`,
                        `print(f"Model loaded successfully: ${model.model_name}")`
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        `## Model Configuration\n`,
                        `\n`,
                        `**Target Variable:** \`${model.target}\`\n`,
                        `\n`,
                        `**Features:** ${model.features.join(', ')}\n`,
                        `\n`,
                        `**Train/Test Split:** ${model.train_test_split * 100}% / ${(1 - model.train_test_split) * 100}%`
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "source": [
                        `# Load your data\n`,
                        `# df = pd.read_csv('your_data.csv')\n`,
                        `\n`,
                        `# Prepare features and target\n`,
                        `# X = df[${JSON.stringify(model.features)}]\n`,
                        `# y = df['${model.target}']\n`,
                        `\n`,
                        `# Make predictions\n`,
                        `# predictions = model.predict(X)\n`,
                        `# print(f"Predictions: {predictions[:5]}")`
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        `## Model Coefficients\n`,
                        `\n`,
                        `Feature importance or coefficients:`
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "source": [
                        `# Display coefficients\n`,
                        `coefficients = ${JSON.stringify(model.coefficients, null, 2)}\n`,
                        `\n`,
                        `for feature, coef in coefficients.items():\n`,
                        `    print(f"{feature}: {coef}")\n`,
                        `\n`,
                        `intercept = ${model.intercept}\n`,
                        `print(f"\\nIntercept: {intercept}")`
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        `## Visualizations`
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "source": [
                        `# Feature importance visualization\n`,
                        `plt.figure(figsize=(10, 6))\n`,
                        `features = list(coefficients.keys())\n`,
                        `values = list(coefficients.values())\n`,
                        `plt.barh(features, values)\n`,
                        `plt.xlabel('Coefficient Value')\n`,
                        `plt.title('Feature Importance')\n`,
                        `plt.tight_layout()\n`,
                        `plt.show()`
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        };
        
        // Create and download the notebook file
        const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${model.model_name}_notebook.ipynb`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        logToConsole(`Notebook exported: ${model.model_name}_notebook.ipynb`, 'success');
    } catch (error) {
        alert(`Error exporting notebook: ${error.message}`);
        logToConsole(`Error exporting notebook: ${error.message}`, 'error');
    }
}
*/

// Notebook Viewer Module
function initializeNotebookModule() {
  const uploadZone = document.getElementById("notebook-upload-zone");
  const fileInput = document.getElementById("notebook-file-input");

  uploadZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) =>
    handleNotebookUpload(e.target.files[0])
  );

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    handleNotebookUpload(e.dataTransfer.files[0]);
  });
}

async function handleNotebookUpload(file) {
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const viewer = document.getElementById("notebook-viewer");
  viewer.innerHTML = '<div class="spinner"></div> Processing notebook...';

  try {
    const response = await fetch("/api/notebook/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      displayNotebook(data);
    } else {
      viewer.innerHTML = `<div class="error-message">${data.error}</div>`;
    }
  } catch (error) {
    viewer.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
  }
}

function displayNotebook(data) {
  const viewer = document.getElementById("notebook-viewer");
  viewer.innerHTML = `<h3>${escapeHtml(data.filename)}</h3>`;

  data.cells.forEach((cell) => {
    const cellDiv = document.createElement("div");
    cellDiv.className = "notebook-cell";

    const cellType =
      cell.type === "code"
        ? "Code"
        : cell.type === "markdown"
        ? "Markdown"
        : "Raw";

    cellDiv.innerHTML = `
            <div class="cell-header">
                <span class="cell-type">${cellType}</span>
                <span>Cell ${cell.index + 1}</span>
            </div>
            <div class="cell-content">${escapeHtml(cell.source)}</div>
            ${
              cell.outputs && cell.outputs.length > 0
                ? `
                <div class="cell-output">
                    <strong>Output:</strong><br>
                    ${cell.outputs
                      .map((out) => escapeHtml(out.data || out.text || ""))
                      .join("<br>")}
                </div>
            `
                : ""
            }
        `;

    viewer.appendChild(cellDiv);
  });
}

// Settings Module
function initializeSettingsModule() {
  // Settings tab navigation
  const settingsNavBtns = document.querySelectorAll(".settings-nav-btn");
  settingsNavBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabName = btn.getAttribute("data-settings-tab");
      switchSettingsTab(tabName);

      // Load storage info when switching to that tab
      if (tabName === "storage") {
        loadStoragePaths();
        loadStorageUsage();
      }
    });
  });

  // Theme cards
  const themeCards = document.querySelectorAll(".theme-card");
  themeCards.forEach((card) => {
    card.addEventListener("click", () => {
      themeCards.forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      const theme = card.getAttribute("data-theme");
      appState.settings.theme = theme;
    });
  });

  // Brightness slider
  const brightnessSlider = document.getElementById("setting-brightness");
  const brightnessValue = document.getElementById("brightness-value");
  if (brightnessSlider && brightnessValue) {
    brightnessSlider.addEventListener("input", (e) => {
      brightnessValue.textContent = `${e.target.value}%`;
    });
  }

  // Chart search functionality
  const chartSearch = document.getElementById("chart-search");
  if (chartSearch) {
    chartSearch.addEventListener("input", (e) => {
      const searchTerm = e.target.value.toLowerCase();
      const chartOptions = document.querySelectorAll(".chart-option");

      chartOptions.forEach((option) => {
        const keywords = option.getAttribute("data-keywords") || "";
        const label =
          option.querySelector("span")?.textContent.toLowerCase() || "";
        const description =
          option
            .querySelector(".setting-description")
            ?.textContent.toLowerCase() || "";

        const matches =
          keywords.includes(searchTerm) ||
          label.includes(searchTerm) ||
          description.includes(searchTerm);

        option.style.display = matches || searchTerm === "" ? "block" : "none";
      });
    });
  }

  // Load current settings into UI
  loadSettingsIntoUI();

  // Save settings button
  const saveBtn = document.getElementById("save-settings-btn");
  if (saveBtn) {
    saveBtn.addEventListener("click", saveSettings);
  }

  // Reset settings button
  const resetBtn = document.getElementById("reset-settings-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", resetSettings);
  }
}

// Removed AI Models Management (feature removed)

function switchSettingsTab(tabName) {
  console.log("Switching to settings tab:", tabName);

  // Update nav buttons
  const navButtons = document.querySelectorAll(".settings-nav-btn");
  navButtons.forEach((btn) => {
    btn.classList.remove("active");
  });

  const targetNavBtn = document.querySelector(
    `[data-settings-tab="${tabName}"]`
  );
  if (targetNavBtn) {
    targetNavBtn.classList.add("active");
  } else {
    console.error("Nav button not found for:", tabName);
  }

  // Update tab content
  const allTabs = document.querySelectorAll(".settings-tab");
  allTabs.forEach((tab) => {
    tab.classList.remove("active");
  });

  const targetTab = document.getElementById(`settings-tab-${tabName}`);
  if (targetTab) {
    targetTab.classList.add("active");
    console.log("Activated tab:", `settings-tab-${tabName}`);
  } else {
    console.error("Tab not found:", `settings-tab-${tabName}`);
  }

  // Load data for specific tabs
  if (tabName === "storage") {
    loadStoragePaths();
    loadStorageUsage();
  }
}

// Storage management functions
async function loadStoragePaths() {
  try {
    const response = await fetch("/api/storage/paths");
    const data = await response.json();

    if (data.success) {
      document.getElementById("path-uploads").textContent = data.paths.uploads;
      document.getElementById("path-models").textContent = data.paths.models;
      document.getElementById("path-cache").textContent = data.paths.cache;
    }
  } catch (error) {
    console.error("Failed to load storage paths:", error);
    logToConsole("Storage: Failed to load paths", "error");
  }
}

async function loadStorageUsage() {
  try {
    const response = await fetch("/api/storage/usage");
    const data = await response.json();

    if (data.success) {
      const usage = data.usage;
      document.getElementById("storage-usage-display").innerHTML = `
                <div class="storage-stat">
                    <strong>Uploads:</strong> ${usage.uploads.size_mb} MB (${usage.uploads.file_count} files)
                </div>
                <div class="storage-stat">
                    <strong>Models:</strong> ${usage.models.size_mb} MB (${usage.models.file_count} files)
                </div>
            `;
    }
  } catch (error) {
    document.getElementById("storage-usage-display").innerHTML =
      '<p class="error-message">Failed to load storage usage</p>';
    logToConsole("Storage: Failed to load usage info", "error");
  }
}

// Clipboard helper
function copyToClipboard(text) {
  navigator.clipboard
    .writeText(text)
    .then(() => {
      showSettingsMessage("Path copied to clipboard!", "success");
    })
    .catch(() => {
      showSettingsMessage("Failed to copy path", "error");
    });
}

function loadSettingsIntoUI() {
  // Visual effects - CRT
  const scanlinesCheckbox = document.getElementById("setting-scanlines");
  const flickerCheckbox = document.getElementById("setting-flicker");
  const glowCheckbox = document.getElementById("setting-glow");
  const vignetteCheckbox = document.getElementById("setting-vignette");
  const curvatureCheckbox = document.getElementById("setting-curvature");
  const brightnessSlider = document.getElementById("setting-brightness");
  const brightnessValue = document.getElementById("brightness-value");

  if (scanlinesCheckbox)
    scanlinesCheckbox.checked = appState.settings.scanlines !== false;
  if (flickerCheckbox)
    flickerCheckbox.checked = appState.settings.flicker !== false;
  if (glowCheckbox) glowCheckbox.checked = appState.settings.glow !== false;
  if (vignetteCheckbox)
    vignetteCheckbox.checked = appState.settings.vignette !== false;
  if (curvatureCheckbox)
    curvatureCheckbox.checked = appState.settings.crtCurvature || false;
  if (brightnessSlider)
    brightnessSlider.value = appState.settings.crtBrightness || 100;
  if (brightnessValue)
    brightnessValue.textContent = `${appState.settings.crtBrightness || 100}%`;

  // System stats
  const systemStatsCheckbox = document.getElementById("setting-system-stats");
  const statsFrequencySelect = document.getElementById(
    "setting-stats-frequency"
  );

  if (systemStatsCheckbox) {
    systemStatsCheckbox.checked = appState.settings.systemStats;
  }
  if (statsFrequencySelect) {
    statsFrequencySelect.value = appState.settings.statsFrequency;
  }

  // GPU settings
  const gpuEnabledCheckbox = document.getElementById("setting-gpu-enabled");
  const gpuForceCpuCheckbox = document.getElementById("setting-gpu-force-cpu");
  const gpuPreferredBackendSelect = document.getElementById(
    "setting-gpu-preferred-backend"
  );

  if (gpuEnabledCheckbox) {
    gpuEnabledCheckbox.checked = appState.settings.gpuEnabled !== false;
  }
  if (gpuForceCpuCheckbox) {
    gpuForceCpuCheckbox.checked = appState.settings.gpuForceCpu || false;
  }
  if (gpuPreferredBackendSelect) {
    gpuPreferredBackendSelect.value =
      appState.settings.gpuPreferredBackend || "auto";
  }

  // Load GPU status
  loadGpuStatus();

  // Theme selection
  document.querySelectorAll(".theme-card").forEach((card) => {
    card.classList.remove("selected");
    if (card.getAttribute("data-theme") === appState.settings.theme) {
      card.classList.add("selected");
    }
  });

  // Training metrics
  if (appState.settings.showMetrics) {
    document.getElementById("metric-r2").checked =
      appState.settings.showMetrics.r2Score !== false;
    document.getElementById("metric-rmse").checked =
      appState.settings.showMetrics.rmse !== false;
    document.getElementById("metric-mae").checked =
      appState.settings.showMetrics.mae !== false;
    document.getElementById("metric-mse").checked =
      appState.settings.showMetrics.mse || false;
    document.getElementById("metric-adjusted-r2").checked =
      appState.settings.showMetrics.adjustedR2 || false;
  }

  // Training charts
  if (appState.settings.showCharts) {
    const charts = appState.settings.showCharts;
    document.getElementById("chart-actual-vs-predicted").checked =
      charts.actualVsPredicted !== false;
    document.getElementById("chart-residuals").checked =
      charts.residualPlot !== false;
    document.getElementById("chart-feature-importance").checked =
      charts.featureImportance !== false;
    document.getElementById("chart-distribution").checked =
      charts.distribution || false;

    // New chart options
    const learningCurveEl = document.getElementById("chart-learning-curve");
    if (learningCurveEl)
      learningCurveEl.checked = charts.learningCurve || false;

    const correlationMatrixEl = document.getElementById(
      "chart-correlation-matrix"
    );
    if (correlationMatrixEl)
      correlationMatrixEl.checked = charts.correlationMatrix || false;

    const confusionMatrixEl = document.getElementById("chart-confusion-matrix");
    if (confusionMatrixEl)
      confusionMatrixEl.checked = charts.confusionMatrix !== false;

    const rocCurveEl = document.getElementById("chart-roc-curve");
    if (rocCurveEl) rocCurveEl.checked = charts.rocCurve !== false;

    const clusterScatterEl = document.getElementById("chart-cluster-scatter");
    if (clusterScatterEl)
      clusterScatterEl.checked = charts.clusterScatter !== false;

    const screePlotEl = document.getElementById("chart-scree-plot");
    if (screePlotEl) screePlotEl.checked = charts.screePlot !== false;

    const transformedScatterEl = document.getElementById(
      "chart-transformed-scatter"
    );
    if (transformedScatterEl)
      transformedScatterEl.checked = charts.transformedScatter !== false;
  }
}

function loadSettings() {
  const saved = localStorage.getItem("mlsuite-settings");
  if (saved) {
    try {
      appState.settings = { ...appState.settings, ...JSON.parse(saved) };
    } catch (e) {
      console.error("Failed to load settings:", e);
    }
  }
}

function applySettings() {
  const body = document.body;

  // Apply visual effects
  body.classList.toggle("scanlines-enabled", appState.settings.scanlines);
  body.classList.toggle("flicker-enabled", appState.settings.flicker);
  body.classList.toggle("glow-enabled", appState.settings.glow);
  body.classList.toggle("vignette-enabled", appState.settings.vignette);
  body.classList.toggle("system-stats-hidden", !appState.settings.systemStats);

  // Apply theme
  body.classList.remove(
    "theme-amber",
    "theme-blue",
    "theme-white",
    "theme-red",
    "theme-purple",
    "theme-matrix",
    "theme-hacker",
    "theme-light"
  );
  if (appState.settings.theme !== "green") {
    body.classList.add(`theme-${appState.settings.theme}`);
  }

  // Apply curvature
  body.classList.toggle(
    "curvature-enabled",
    appState.settings.crtCurvature || false
  );

  // Apply brightness
  document.documentElement.style.setProperty(
    "--screen-brightness",
    `${appState.settings.crtBrightness || 100}%`
  );
}

function saveSettings() {
  // Get values from UI - Visual effects (CRT)
  const scanlinesCheckbox = document.getElementById("setting-scanlines");
  const flickerCheckbox = document.getElementById("setting-flicker");
  const glowCheckbox = document.getElementById("setting-glow");
  const vignetteCheckbox = document.getElementById("setting-vignette");
  const curvatureCheckbox = document.getElementById("setting-curvature");
  const brightnessSlider = document.getElementById("setting-brightness");

  if (scanlinesCheckbox)
    appState.settings.scanlines = scanlinesCheckbox.checked;
  if (flickerCheckbox) appState.settings.flicker = flickerCheckbox.checked;
  if (glowCheckbox) appState.settings.glow = glowCheckbox.checked;
  if (vignetteCheckbox) appState.settings.vignette = vignetteCheckbox.checked;
  if (curvatureCheckbox)
    appState.settings.crtCurvature = curvatureCheckbox.checked;
  if (brightnessSlider)
    appState.settings.crtBrightness = parseInt(brightnessSlider.value);

  // System stats
  const systemStatsCheckbox = document.getElementById("setting-system-stats");
  const statsFrequencySelect = document.getElementById(
    "setting-stats-frequency"
  );

  if (systemStatsCheckbox) {
    appState.settings.systemStats = systemStatsCheckbox.checked;
  }
  if (statsFrequencySelect) {
    appState.settings.statsFrequency = parseInt(statsFrequencySelect.value);
  }

  // GPU settings
  const gpuEnabledCheckbox = document.getElementById("setting-gpu-enabled");
  const gpuForceCpuCheckbox = document.getElementById("setting-gpu-force-cpu");
  const gpuPreferredBackendSelect = document.getElementById(
    "setting-gpu-preferred-backend"
  );

  if (gpuEnabledCheckbox) {
    appState.settings.gpuEnabled = gpuEnabledCheckbox.checked;
  }
  if (gpuForceCpuCheckbox) {
    appState.settings.gpuForceCpu = gpuForceCpuCheckbox.checked;
  }
  if (gpuPreferredBackendSelect) {
    appState.settings.gpuPreferredBackend = gpuPreferredBackendSelect.value;
  }

  // Save GPU settings to backend
  saveGpuSettingsToBackend();

  // Theme (already set by card click)

  // Training metrics
  appState.settings.showMetrics = {
    r2Score: document.getElementById("metric-r2").checked,
    rmse: document.getElementById("metric-rmse").checked,
    mae: document.getElementById("metric-mae").checked,
    mse: document.getElementById("metric-mse").checked,
    adjustedR2: document.getElementById("metric-adjusted-r2").checked,
  };

  // Training charts
  appState.settings.showCharts = {
    actualVsPredicted: document.getElementById("chart-actual-vs-predicted")
      .checked,
    residualPlot: document.getElementById("chart-residuals").checked,
    featureImportance: document.getElementById("chart-feature-importance")
      .checked,
    distribution: document.getElementById("chart-distribution").checked,
    learningCurve:
      document.getElementById("chart-learning-curve")?.checked || false,
    correlationMatrix:
      document.getElementById("chart-correlation-matrix")?.checked || false,
    confusionMatrix:
      document.getElementById("chart-confusion-matrix")?.checked || true,
    rocCurve: document.getElementById("chart-roc-curve")?.checked || true,
    clusterScatter:
      document.getElementById("chart-cluster-scatter")?.checked || true,
    screePlot: document.getElementById("chart-scree-plot")?.checked || true,
    transformedScatter:
      document.getElementById("chart-transformed-scatter")?.checked || true,
  };

  // Save to localStorage
  localStorage.setItem("mlsuite-settings", JSON.stringify(appState.settings));

  // Apply settings
  applySettings();

  // Restart system stats with new frequency
  restartSystemStats();

  // Show success message
  showSettingsMessage("Settings saved successfully!", "success");
  logToConsole("Settings saved and applied", "success");
}

function resetSettings() {
  if (!confirm("Reset all settings to defaults?")) {
    return;
  }

  // Reset to defaults
  appState.settings = {
    scanlines: false,
    flicker: false,
    glow: false,
    vignette: true,
    systemStats: true,
    statsFrequency: 2000,
    theme: "amber",
    crtCurvature: false,
    crtBrightness: 100,
    showMetrics: {
      r2Score: true,
      rmse: true,
      mae: true,
      mse: false,
      adjustedR2: false,
    },
    showCharts: {
      actualVsPredicted: true,
      residualPlot: true,
      featureImportance: true,
      distribution: false,
      learningCurve: false,
      correlationMatrix: false,
      confusionMatrix: true,
      rocCurve: true,
      clusterScatter: true,
      screePlot: true,
      transformedScatter: true,
    },
  };

  // Update UI
  loadSettingsIntoUI();

  // Save and apply
  localStorage.setItem("mlsuite-settings", JSON.stringify(appState.settings));
  applySettings();
  restartSystemStats();

  showSettingsMessage("Settings reset to defaults", "success");
  logToConsole("Settings reset to defaults", "info");
}

function showSettingsMessage(message, type) {
  const existingAlert = document.querySelector(".settings-alert");
  if (existingAlert) {
    existingAlert.remove();
  }

  const alert = document.createElement("div");
  alert.className = `settings-alert ${type}`;
  alert.textContent = message;

  const settingsContent = document.querySelector(
    "#settings-module .module-content"
  );
  settingsContent.insertBefore(alert, settingsContent.firstChild);

  setTimeout(() => alert.remove(), 3000);
}

// Utility Functions
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function copyToClipboard(text) {
  navigator.clipboard
    .writeText(text)
    .then(() => {
      alert("Copied to clipboard!");
    })
    .catch((err) => {
      console.error("Failed to copy:", err);
    });
}

// GPU Settings Functions
async function loadGpuStatus() {
  try {
    const response = await fetch("/api/system/stats");
    const data = await response.json();

    const statusElement = document.getElementById("gpu-status-text");
    if (!statusElement) return;

    if (data.gpu && data.gpu.available) {
      let statusText = `✓ Available (${data.gpu.gpu_backend.toUpperCase()})`;
      if (data.gpu.devices && data.gpu.devices.length > 0) {
        const device = data.gpu.devices[0];
        if (device.type === "cuda") {
          statusText += ` - ${device.name}`;
        } else if (device.type === "mps") {
          statusText += ` - Apple Silicon`;
        }
      }
      statusElement.textContent = statusText;
      statusElement.style.color = "#00ff00";
    } else {
      statusElement.textContent = "✗ Not Available (CPU Only)";
      statusElement.style.color = "#ffaa00";
    }
  } catch (error) {
    console.error("Failed to load GPU status:", error);
    const statusElement = document.getElementById("gpu-status-text");
    if (statusElement) {
      statusElement.textContent = "Error checking GPU status";
      statusElement.style.color = "#ff0000";
    }
  }
}

async function saveGpuSettingsToBackend() {
  try {
    const settings = {
      gpu_enabled: appState.settings.gpuEnabled !== false,
      gpu_force_cpu: appState.settings.gpuForceCpu || false,
      gpu_preferred_backend: appState.settings.gpuPreferredBackend || "auto",
    };

    const response = await fetch("/api/system/gpu/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });

    const data = await response.json();

    if (data.success) {
      console.log("GPU settings saved:", data.settings);
      // Reload GPU status after saving
      setTimeout(() => loadGpuStatus(), 500);
    } else {
      console.error("Failed to save GPU settings:", data.error);
    }
  } catch (error) {
    console.error("Error saving GPU settings:", error);
  }
}
