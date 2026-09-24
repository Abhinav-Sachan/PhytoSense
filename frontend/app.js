/**
 * app.js
 * ------
 * No login, no API key, no setup step: the frontend talks straight to
 * the local Flask backend at BACKEND_URL. Just open this page with
 * backend/main.py running and it works.
 *
 * The LLM key itself never appears here — /api/ai/* routes are
 * handled entirely by the backend, which holds the real Anthropic key
 * in its own .env file (see backend/llm_service.py).
 */

const BACKEND_URL = "http://127.0.0.1:5000";
const POLL_INTERVAL_MS = 1000;

const state = {
  connected: false,
  pollHandle: null,
  traceMode: "filtered",
  currentView: "home",
  chatHistory: [],
};

const el = {
  navLinks: document.querySelectorAll(".nav-link"),
  navButtons: document.querySelectorAll("[data-nav]"),
  views: document.querySelectorAll(".view"),
  connectionStatus: document.getElementById("connectionStatus"),

  classLabel: document.getElementById("classLabel"),
  classConfidence: document.getElementById("classConfidence"),
  probBars: document.getElementById("probBars"),
  valRms: document.getElementById("valRms"),
  valPeak: document.getElementById("valPeak"),
  valFreq: document.getElementById("valFreq"),
  valZcr: document.getElementById("valZcr"),
  logBody: document.getElementById("logBody"),

  runDiagnosisBtn: document.getElementById("runDiagnosisBtn"),
  statusBody: document.getElementById("statusBody"),
  statusMeta: document.getElementById("statusMeta"),

  chatMessages: document.getElementById("chatMessages"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
};

// Navigation — plain view switching, no gating
function navigateTo(view) {
  state.currentView = view;
  el.views.forEach((v) => { v.hidden = v.dataset.view !== view; });
  el.navLinks.forEach((btn) => btn.classList.toggle("active", btn.dataset.view === view));
}
el.navLinks.forEach((btn) => btn.addEventListener("click", () => navigateTo(btn.dataset.view)));
el.navButtons.forEach((btn) => btn.addEventListener("click", () => navigateTo(btn.dataset.nav)));

function setConnectionState(online) {
  state.connected = online;
  el.connectionStatus.dataset.state = online ? "online" : "offline";
  el.connectionStatus.querySelector(".status-text").textContent = online ? "Connected" : "Backend offline";
}

// Networking helpers
async function apiGet(path) {
  const res = await fetch(`${BACKEND_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${res.status})`);
  }
  return res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${res.status})`);
  }
  return res.json();
}

// Charts
const CHART_GREEN = "#7fb88f";
const CHART_AMBER = "#dba55b";
const GRID_COLOR = "rgba(255,255,255,0.05)";
const TICK_COLOR = "#5c675f";

Chart.defaults.font.family = "IBM Plex Mono";
Chart.defaults.font.size = 11;

const waveformChart = new Chart(document.getElementById("waveformChart"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      { label: "Filtered", data: [], borderColor: CHART_GREEN, borderWidth: 1.5, pointRadius: 0, tension: 0.15 },
      { label: "Raw", data: [], borderColor: CHART_AMBER, borderWidth: 1, pointRadius: 0, tension: 0.1, hidden: true },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR } },
    },
  },
});

const spectrumChart = new Chart(document.getElementById("spectrumChart"), {
  type: "bar",
  data: { labels: [], datasets: [{ label: "Magnitude", data: [], backgroundColor: CHART_GREEN }] },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false }, ticks: { color: TICK_COLOR, maxTicksLimit: 8 }, title: { display: true, text: "Hz", color: TICK_COLOR } },
      y: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR } },
    },
  },
});

document.querySelectorAll(".toggle-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.traceMode = btn.dataset.trace;
    const [filteredSet, rawSet] = waveformChart.data.datasets;
    filteredSet.hidden = state.traceMode === "raw";
    rawSet.hidden = state.traceMode === "filtered";
    waveformChart.update();
  });
});

// Polling: waveform + metrics
let lastLoggedLabel = null;

async function pollWaveform() {
  const data = await apiGet("/api/waveform");
  waveformChart.data.labels = data.filtered.map((_, i) => i);
  waveformChart.data.datasets[0].data = data.filtered;
  waveformChart.data.datasets[1].data = data.raw;
  waveformChart.update();
}

async function pollMetrics() {
  const data = await apiGet("/api/metrics");
  if (!data.ready) return;

  el.classLabel.textContent = data.label;
  el.classConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
  renderProbBars(data.probabilities);

  el.valRms.textContent = data.features.rms.toFixed(3);
  el.valPeak.textContent = data.features.max_amplitude.toFixed(3);
  el.valFreq.textContent = `${data.features.dominant_frequency.toFixed(1)} Hz`;
  el.valZcr.textContent = data.features.zero_crossing_rate.toFixed(3);

  renderSpectrumFromFeatures(data);

  if (data.label !== lastLoggedLabel) {
    appendLogRow(data.label, data.confidence);
    lastLoggedLabel = data.label;
  }
}

function renderProbBars(probabilities) {
  el.probBars.innerHTML = "";
  Object.entries(probabilities).sort((a, b) => b[1] - a[1]).forEach(([label, prob]) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span>${label}</span>
      <span class="prob-track"><span class="prob-fill" style="width:${(prob * 100).toFixed(0)}%"></span></span>
      <span>${(prob * 100).toFixed(0)}%</span>`;
    el.probBars.appendChild(row);
  });
}

function appendLogRow(label, confidence) {
  const row = document.createElement("tr");
  const time = new Date().toLocaleTimeString("en-GB", { hour12: false });
  row.innerHTML = `<td>${time}</td><td>${label}</td><td>${(confidence * 100).toFixed(0)}%</td>`;
  el.logBody.prepend(row);
  while (el.logBody.rows.length > 30) el.logBody.deleteRow(-1);
}

function renderSpectrumFromFeatures(data) {
  const dominant = data.features.dominant_frequency;
  const bins = Array.from({ length: 20 }, (_, i) => (i * 2.5).toFixed(1));
  const magnitudes = bins.map((freqStr) => {
    const distance = Math.abs(parseFloat(freqStr) - dominant);
    return Math.max(0, 1 - distance / 8) * (0.5 + data.confidence);
  });
  spectrumChart.data.labels = bins;
  spectrumChart.data.datasets[0].data = magnitudes;
  spectrumChart.update();
}

async function pollLoop() {
  try {
    await Promise.all([pollWaveform(), pollMetrics()]);
    setConnectionState(true);
  } catch (err) {
    setConnectionState(false);
  }
}

function startPolling() {
  if (state.pollHandle) clearInterval(state.pollHandle);
  pollLoop();
  state.pollHandle = setInterval(pollLoop, POLL_INTERVAL_MS);
}

// AI: status diagnosis
el.runDiagnosisBtn.addEventListener("click", async () => {
  el.runDiagnosisBtn.disabled = true;
  el.runDiagnosisBtn.textContent = "Analyzing…";
  el.statusBody.innerHTML = `<p class="status-placeholder">Reading current sensor data…</p>`;

  try {
    const result = await apiPost("/api/ai/status", {});
    el.statusBody.innerHTML = `<p class="status-text"></p>`;
    el.statusBody.querySelector(".status-text").textContent = result.status_text;
    el.statusMeta.textContent = `Based on: ${result.based_on.label} (${(result.based_on.confidence * 100).toFixed(0)}% confidence) — generated ${new Date().toLocaleTimeString("en-GB", { hour12: false })}`;
  } catch (err) {
    el.statusBody.innerHTML = `<p class="status-placeholder">${err.message}</p>`;
  } finally {
    el.runDiagnosisBtn.disabled = false;
    el.runDiagnosisBtn.textContent = "Analyze now";
  }
});

// AI: chat
function appendChatMessage(role, content) {
  const wrap = document.createElement("div");
  wrap.className = `chat-msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.textContent = content;
  wrap.appendChild(bubble);
  el.chatMessages.appendChild(wrap);
  el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  return bubble;
}

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = el.chatInput.value.trim();
  if (!message) return;

  appendChatMessage("user", message);
  state.chatHistory.push({ role: "user", content: message });
  el.chatInput.value = "";

  const thinkingBubble = appendChatMessage("assistant", "Thinking…");

  try {
    const result = await apiPost("/api/ai/chat", { message, history: state.chatHistory });
    thinkingBubble.textContent = result.reply;
    state.chatHistory.push({ role: "assistant", content: result.reply });
  } catch (err) {
    thinkingBubble.textContent = `Couldn't get a response: ${err.message}`;
  }
});

el.chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    el.chatForm.requestSubmit();
  }
});

// Boot — start polling immediately, no connect step
startPolling();