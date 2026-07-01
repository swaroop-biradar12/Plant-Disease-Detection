const API_URL = "http://127.0.0.1:8000";

const panels = {
  upload: document.getElementById("panel-upload"),
  scan: document.getElementById("panel-scan"),
  results: document.getElementById("panel-results"),
  error: document.getElementById("panel-error"),
};

function showPanel(name) {
  Object.values(panels).forEach(p => p.dataset.active = "false");
  panels[name].dataset.active = "true";
}

// ── Backend health check ──────────────────────────────────────────────────
async function checkBackend() {
  const dot = document.getElementById("backendDot");
  const status = document.getElementById("backendStatus");
  try {
    const res = await fetch(`${API_URL}/`);
    const data = await res.json();
    if (data.model_loaded) {
      dot.classList.add("online");
      status.textContent = "backend online · model loaded";
    } else {
      dot.classList.add("offline");
      status.textContent = "backend online · model missing";
    }
  } catch {
    dot.classList.add("offline");
    status.textContent = "backend unreachable";
  }
}
checkBackend();

// ── Confidence slider ─────────────────────────────────────────────────────
const confSlider = document.getElementById("confSlider");
const confVal = document.getElementById("confVal");
confSlider.addEventListener("input", () => {
  confVal.textContent = parseFloat(confSlider.value).toFixed(2);
});

// ── Upload handling ───────────────────────────────────────────────────────
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

["dragenter", "dragover"].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

let currentFile = null;

function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById("scanImage").src = e.target.result;
    document.getElementById("resultImage").src = e.target.result;
    runDetection();
  };
  reader.readAsDataURL(file);
}

// ── Scan log animation ────────────────────────────────────────────────────
function setLogStep(step) {
  const lines = document.querySelectorAll(".scan-log-line");
  lines.forEach(line => {
    if (line.dataset.step === step) {
      line.classList.add("active");
      line.classList.remove("complete");
    } else if ([...lines].findIndex(l => l.dataset.step === step) > [...lines].indexOf(line)) {
      line.classList.add("complete");
      line.classList.remove("active");
    }
  });
}

// ── Run detection ─────────────────────────────────────────────────────────
async function runDetection() {
  showPanel("scan");
  setLogStep("detect");

  const apiKey = document.getElementById("apiKey").value.trim();
  const language = document.getElementById("langSelect").value;
  const confidence = confSlider.value;

  const formData = new FormData();
  formData.append("file", currentFile);
  formData.append("confidence", confidence);
  formData.append("gemini_key", apiKey);
  formData.append("language", language);

  try {
    setTimeout(() => setLogStep("advice"), 800);

    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error(`Server responded ${res.status}`);

    const data = await res.json();
    console.log("Backend response:", data);
    setLogStep("done");

    setTimeout(() => renderResults(data), 500);

  } catch (err) {
    showError(err.message || "Could not reach the backend. Make sure FastAPI is running on port 8000.");
  }
}

// ── Render results ────────────────────────────────────────────────────────
let lastReportText = "";

function renderResults(data) {
  const detectionsList = document.getElementById("detectionsList");
  detectionsList.innerHTML = "";

  if (data.healthy) {
    detectionsList.innerHTML = `
      <div class="detection-row healthy">
        <span class="detection-name"><i class="ti ti-check" aria-hidden="true"></i>&nbsp; No disease detected</span>
      </div>`;
  } else {
    data.detections.forEach(d => {
      const pct = Math.round(d.confidence * 100);
      detectionsList.innerHTML += `
        <div class="detection-row">
          <span class="detection-name">${d.disease}</span>
          <span class="detection-conf">${pct}%</span>
        </div>`;
    });
  }

  const reportBody = document.getElementById("reportBody");
  const listenBtn = document.getElementById("listenBtn");

  if (data.ai_advice && data.ai_advice.startsWith("Gemini error")) {
    const isQuota = data.ai_advice.includes("429") || data.ai_advice.includes("RESOURCE_EXHAUSTED");
    reportBody.innerHTML = `
      <p style="color: var(--amber)">
        ${isQuota
          ? "Gemini's free daily quota is used up for this API key. Wait a while or use a different key."
          : "Gemini could not generate a report right now. Try again shortly."}
      </p>`;
    listenBtn.style.display = "none";
  } else if (data.ai_advice) {
    const parsed = parseReport(data.ai_advice);
    lastReportText = buildSpeechText(parsed);

    // Reset button visibility — it may have been hidden by a previous error
    listenBtn.style.display = "inline-flex";

    reportBody.innerHTML = `
      <div class="report-row">
        <p>${parsed.OVERVIEW || ""}</p>
      </div>
      <div class="report-stats">
        <div>
          <p class="report-stat-label">Severity</p>
          <p class="report-stat-value">${parsed.SEVERITY || "—"}</p>
        </div>
        <div>
          <p class="report-stat-label">Likely cause</p>
          <p class="report-stat-value">${parsed.CAUSE || "—"}</p>
        </div>
      </div>
      <div class="report-row">
        <p class="report-label">Symptoms</p>
        <p>${parsed.SYMPTOMS || ""}</p>
      </div>
      <div class="report-row">
        <p class="report-label">Treatment</p>
        <ul class="report-list">${listItems(parsed.TREATMENT)}</ul>
      </div>
      <div class="report-row">
        <p class="report-label">Prevention</p>
        <ul class="report-list">${listItems(parsed.PREVENTION)}</ul>
      </div>
    `;
  } else {
    reportBody.innerHTML = `<p style="color: var(--text-faint)">Enter a Gemini API key to generate a full diagnosis report.</p>`;
    listenBtn.style.display = "none";
  }

  showPanel("results");
}

function parseReport(text) {
  const result = {};
  text.split("\n").forEach(line => {
    const idx = line.indexOf(":");
    if (idx > -1) {
      const key = line.slice(0, idx).trim().toUpperCase();
      const value = line.slice(idx + 1).trim();
      result[key] = value;
    }
  });
  return result;
}

function listItems(pipeStr) {
  if (!pipeStr) return "";
  return pipeStr.split("|").map(item => `<li>${item.trim()}</li>`).join("");
}

function buildSpeechText(parsed) {
  const treatment = (parsed.TREATMENT || "").split("|").join(". ");
  const prevention = (parsed.PREVENTION || "").split("|").join(". ");
  return `${parsed.OVERVIEW || ""} Severity: ${parsed.SEVERITY || ""}. Likely cause: ${parsed.CAUSE || ""}. Symptoms: ${parsed.SYMPTOMS || ""}. Treatment steps: ${treatment}. Prevention tips: ${prevention}.`;
}

// ── Listen to report ──────────────────────────────────────────────────────
document.getElementById("listenBtn").addEventListener("click", async () => {
  const btn = document.getElementById("listenBtn");
  const player = document.getElementById("audioPlayer");
  const language = document.getElementById("langSelect").value;

  if (!lastReportText) return;

  btn.disabled = true;
  btn.querySelector("span").textContent = "Generating audio…";

  try {
    const formData = new FormData();
    formData.append("text", lastReportText);
    formData.append("language", language);

    const res = await fetch(`${API_URL}/speak`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Audio generation failed");

    const blob = await res.blob();
    player.src = URL.createObjectURL(blob);
    player.hidden = false;
    player.play();

  } catch (err) {
    alert("Could not generate audio: " + err.message);
  } finally {
    btn.disabled = false;
    btn.querySelector("span").textContent = "Listen to report";
  }
});

// ── Reset / error handling ────────────────────────────────────────────────
function showError(message) {
  document.getElementById("errorDetail").textContent = message;
  showPanel("error");
}

function resetFlow() {
  currentFile = null;
  fileInput.value = "";
  document.getElementById("audioPlayer").hidden = true;
  showPanel("upload");
}

document.getElementById("resetBtn").addEventListener("click", resetFlow);
document.getElementById("errorResetBtn").addEventListener("click", resetFlow);