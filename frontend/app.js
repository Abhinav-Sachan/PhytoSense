/**
 * PhytoSense — Frontend Application
 * ---------------------------------
 * Connects the dashboard to the local Flask backend.
 *
 * Chart.js is loaded locally from:
 *     chart.umd.min.js
 *
 * Data flow:
 *     Flask API → Waveform → Chart.js
 *               → Metrics → Dashboard
 *               → ML Classification
 *               → AI Diagnosis / Chat
 */

const BACKEND_URL = "http://127.0.0.1:5000";
const POLL_INTERVAL_MS = 1000;
/* Samples per second of the waveform the backend sends:
   real ESP32 = 10 per second, built-in simulator = 100 per second */
const SAMPLE_RATE_DEVICE = 10;
const SAMPLE_RATE_SIMULATED = 100;


/* ============================================================
   APPLICATION STATE
   ============================================================ */

const state = {
    connected: false,
    pollHandle: null,
    traceMode: "filtered",
    currentView: "home",
    source: null,          // "device" (real ESP32) or "simulated"
    chatHistory: []
};


/* ============================================================
   DOM ELEMENTS
   ============================================================ */

const el = {
    navLinks: document.querySelectorAll(".nav-link"),
    navButtons: document.querySelectorAll("[data-nav]"),
    views: document.querySelectorAll(".view"),

    connectionStatus:
        document.getElementById("connectionStatus"),

    classLabel:
        document.getElementById("classLabel"),

    classConfidence:
        document.getElementById("classConfidence"),

    probBars:
        document.getElementById("probBars"),

    valRms:
        document.getElementById("valRms"),

    valPeak:
        document.getElementById("valPeak"),

    valFreq:
        document.getElementById("valFreq"),

    valZcr:
        document.getElementById("valZcr"),

    valTemp:
        document.getElementById("valTemp"),

    valHumidity:
        document.getElementById("valHumidity"),

    valLight:
        document.getElementById("valLight"),

    valElectrodes:
        document.getElementById("valElectrodes"),

    logBody:
        document.getElementById("logBody"),

    runDiagnosisBtn:
        document.getElementById("runDiagnosisBtn"),

    statusBody:
        document.getElementById("statusBody"),

    statusMeta:
        document.getElementById("statusMeta"),

    chatMessages:
        document.getElementById("chatMessages"),

    chatForm:
        document.getElementById("chatForm"),

    chatInput:
        document.getElementById("chatInput")
};


/* ============================================================
   NAVIGATION
   ============================================================ */

function navigateTo(view) {

    state.currentView = view;

    el.views.forEach((element) => {
        element.hidden =
            element.dataset.view !== view;
    });

    el.navLinks.forEach((button) => {
        button.classList.toggle(
            "active",
            button.dataset.view === view
        );
    });
}


el.navLinks.forEach((button) => {

    button.addEventListener("click", () => {
        navigateTo(button.dataset.view);
    });

});


el.navButtons.forEach((button) => {

    button.addEventListener("click", () => {
        navigateTo(button.dataset.nav);
    });

});


/* ============================================================
   CONNECTION STATUS
   ============================================================ */

function setConnectionState(online) {

    state.connected = online;

    if (!el.connectionStatus) {
        return;
    }

    /* 3 colours: green = real ESP32 data,
       amber = backend running but ESP32 silent (or simulator on),
       red   = backend (main.py) not running */
    el.connectionStatus.dataset.state =
        !online
            ? "offline"
            : state.source === "device"
                ? "online"
                : "waiting";

    const statusText =
        el.connectionStatus.querySelector(".status-text");

    if (statusText) {

        statusText.textContent =
            !online
                ? "Backend offline"
                : state.source === "device"
                    ? "Live: ESP32"
                    : state.source === "simulated"
                        ? "Simulated data"
                        : "ESP32 offline";
    }
}


/* ============================================================
   API
   ============================================================ */

async function apiGet(path) {

    const response = await fetch(
        `${BACKEND_URL}${path}`
    );

    if (!response.ok) {

        const body =
            await response
                .json()
                .catch(() => ({}));

        throw new Error(
            body.error ||
            `Request failed (${response.status})`
        );
    }

    return response.json();
}


async function apiPost(path, payload) {

    const response = await fetch(
        `${BACKEND_URL}${path}`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
        }
    );

    if (!response.ok) {

        const body =
            await response
                .json()
                .catch(() => ({}));

        throw new Error(
            body.error ||
            `Request failed (${response.status})`
        );
    }

    return response.json();
}


/* ============================================================
   CHART CONFIGURATION
   ============================================================ */

const CHART_GREEN = "#7fb88f";
const CHART_AMBER = "#dba55b";
const GRID_COLOR = "rgba(255,255,255,0.05)";
const TICK_COLOR = "#5c675f";


/* ============================================================
   CHART.JS CHECK
   ============================================================ */

if (typeof Chart === "undefined") {

    throw new Error(
        "Chart.js failed to load. Check chart.umd.min.js in the frontend folder."
    );
}


Chart.defaults.font.family = "IBM Plex Mono";
Chart.defaults.font.size = 11;


/* ============================================================
   WAVEFORM CHART
   ============================================================ */

const waveformCanvas =
    document.getElementById("waveformChart");


const waveformChart =
    new Chart(
        waveformCanvas,
        {
            type: "line",

            data: {

                labels: [],

                datasets: [

                    {
                        label: "Filtered",

                        data: [],

                        borderColor:
                            CHART_GREEN,

                        borderWidth: 1.5,

                        pointRadius: 0,

                        tension: 0.15
                    },

                    {
                        label: "Raw",

                        data: [],

                        borderColor:
                            CHART_AMBER,

                        borderWidth: 1,

                        pointRadius: 0,

                        tension: 0.1,

                        hidden: true
                    }
                ]
            },

            options: {

                animation: false,

                responsive: true,

                maintainAspectRatio: false,

                interaction: {
                    intersect: false,
                    mode: "index"
                },

                plugins: {

                    legend: {
                        display: false
                    },

                    tooltip: {
                        enabled: false
                    }
                },

                scales: {

                    x: {
                        display: false
                    },

                    y: {

                        grid: {
                            color: GRID_COLOR
                        },

                        ticks: {
                            color: TICK_COLOR
                        }
                    }
                }
            }
        }
    );


/* ============================================================
   SPECTRUM CHART
   ============================================================ */

const spectrumCanvas =
    document.getElementById("spectrumChart");


const spectrumChart =
    new Chart(
        spectrumCanvas,
        {
            type: "bar",

            data: {

                labels: [],

                datasets: [

                    {
                        label: "Magnitude",

                        data: [],

                        backgroundColor:
                            CHART_GREEN
                    }
                ]
            },

            options: {

                animation: false,

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        display: false
                    }
                },

                scales: {

                    x: {

                        grid: {
                            display: false
                        },

                        ticks: {

                            color:
                                TICK_COLOR,

                            maxTicksLimit: 8
                        },

                        title: {

                            display: true,

                            text:
                                "Frequency (Hz)",

                            color:
                                TICK_COLOR
                        }
                    },

                    y: {

                        grid: {
                            color:
                                GRID_COLOR
                        },

                        ticks: {
                            color:
                                TICK_COLOR
                        }
                    }
                }
            }
        }
    );


/* ============================================================
   WAVEFORM TOGGLE
   ============================================================ */

document
    .querySelectorAll(".toggle-btn")
    .forEach((button) => {

        button.addEventListener(
            "click",
            () => {

                document
                    .querySelectorAll(".toggle-btn")
                    .forEach((item) => {
                        item.classList.remove(
                            "active"
                        );
                    });

                button.classList.add("active");

                state.traceMode =
                    button.dataset.trace;

                const filteredDataset =
                    waveformChart
                        .data
                        .datasets[0];

                const rawDataset =
                    waveformChart
                        .data
                        .datasets[1];

                filteredDataset.hidden =
                    state.traceMode === "raw";

                rawDataset.hidden =
                    state.traceMode === "filtered";

                waveformChart.update();
            }
        );

    });


/* ============================================================
   REAL FFT CALCULATION
   ============================================================ */

function calculateSpectrum(signal, sampleRate) {

    if (!signal || signal.length < 4) {

        return {
            frequencies: [],
            magnitudes: []
        };
    }

    const originalLength =
        signal.length;

    let n = 1;

    while (n < originalLength) {
        n *= 2;
    }


    const real =
        new Array(n).fill(0);

    const imag =
        new Array(n).fill(0);


    for (
        let i = 0;
        i < originalLength;
        i++
    ) {

        real[i] =
            Number(signal[i]) || 0;
    }


    /* Bit-reversal permutation */

    let j = 0;

    for (
        let i = 1;
        i < n;
        i++
    ) {

        let bit = n >> 1;

        while (j & bit) {

            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j) {

            [
                real[i],
                real[j]
            ] =
            [
                real[j],
                real[i]
            ];
        }
    }


    /* Cooley-Tukey FFT */

    for (
        let length = 2;
        length <= n;
        length *= 2
    ) {

        const angle =
            (-2 * Math.PI) / length;

        const wReal =
            Math.cos(angle);

        const wImag =
            Math.sin(angle);


        for (
            let start = 0;
            start < n;
            start += length
        ) {

            let currentReal = 1;
            let currentImag = 0;

            const half =
                length / 2;


            for (
                let i = 0;
                i < half;
                i++
            ) {

                const evenIndex =
                    start + i;

                const oddIndex =
                    start + i + half;


                const oddReal =
                    real[oddIndex];

                const oddImag =
                    imag[oddIndex];


                const multipliedReal =
                    currentReal * oddReal -
                    currentImag * oddImag;

                const multipliedImag =
                    currentReal * oddImag +
                    currentImag * oddReal;


                real[oddIndex] =
                    real[evenIndex] -
                    multipliedReal;

                imag[oddIndex] =
                    imag[evenIndex] -
                    multipliedImag;


                real[evenIndex] +=
                    multipliedReal;

                imag[evenIndex] +=
                    multipliedImag;


                const nextReal =
                    currentReal * wReal -
                    currentImag * wImag;

                const nextImag =
                    currentReal * wImag +
                    currentImag * wReal;


                currentReal =
                    nextReal;

                currentImag =
                    nextImag;
            }
        }
    }


    /* One-sided spectrum */

    const frequencies = [];
    const magnitudes = [];


    for (
        let i = 0;
        i <= n / 2;
        i++
    ) {

        const frequency =
            (i * sampleRate) / n;

        const magnitude =
            Math.sqrt(
                real[i] * real[i] +
                imag[i] * imag[i]
            ) / n;


        frequencies.push(
            frequency.toFixed(1)
        );

        magnitudes.push(
            magnitude
        );
    }


    return {
        frequencies,
        magnitudes
    };
}


/* ============================================================
   UPDATE SPECTRUM
   ============================================================ */

function renderSpectrum(signal) {

    const sampleRate =
        state.source === "device"
            ? SAMPLE_RATE_DEVICE
            : SAMPLE_RATE_SIMULATED;

    const spectrum =
        calculateSpectrum(signal, sampleRate);

    spectrumChart.data.labels =
        spectrum.frequencies;

    spectrumChart
        .data
        .datasets[0]
        .data =
        spectrum.magnitudes;

    spectrumChart.update("none");
}


/* ============================================================
   WAVEFORM POLLING
   ============================================================ */

async function pollWaveform() {

    const data =
        await apiGet("/api/waveform");

    if (!data) {
        return;
    }


    /* Remember where this data came from (real ESP32 or simulator) */
    state.source = data.source;


    const filtered =
        data.filtered || [];

    const raw =
        data.raw || [];


    waveformChart.data.labels =
        filtered.map(
            (_, index) => index
        );


    waveformChart
        .data
        .datasets[0]
        .data =
        filtered;


    waveformChart
        .data
        .datasets[1]
        .data =
        raw;


    waveformChart.update("none");


    if (filtered.length > 0) {
        renderSpectrum(filtered);
    } else {
        /* No data (ESP32 offline): empty the spectrum too */
        spectrumChart.data.labels = [];
        spectrumChart.data.datasets[0].data = [];
        spectrumChart.update("none");
    }
}


/* ============================================================
   METRICS POLLING
   ============================================================ */

let lastLoggedLabel = null;


async function pollMetrics() {

    const data =
        await apiGet("/api/metrics");

    if (!data) {
        return;
    }

    if (!data.ready) {
        /* Don't keep showing an old result when no data is coming in */
        el.classLabel.textContent = "—";
        el.classConfidence.textContent =
            data.source === "offline"
                ? "Waiting for ESP32…"
                : "Collecting data…";
        renderProbBars({});
        el.valRms.textContent = "—";
        el.valPeak.textContent = "—";
        el.valFreq.textContent = "—";
        el.valZcr.textContent = "—";
        lastLoggedLabel = null;
        return;
    }


    /* Classification */

    el.classLabel.textContent =
        data.label || "Unknown";


    el.classConfidence.textContent =
        `${(
            (data.confidence || 0) * 100
        ).toFixed(1)}% confidence`;


    /* Probability bars */

    renderProbBars(
        data.probabilities || {}
    );


    /* ML features */

    const features =
        data.features || {};


    el.valRms.textContent =
        Number(
            features.rms || 0
        ).toFixed(3);


    el.valPeak.textContent =
        Number(
            features.max_amplitude || 0
        ).toFixed(3);


    el.valFreq.textContent =
        `${Number(
            features.dominant_frequency || 0
        ).toFixed(1)} Hz`;


    el.valZcr.textContent =
        Number(
            features.zero_crossing_rate || 0
        ).toFixed(3);


    /* Session log */

    if (
        data.label &&
        data.label !== lastLoggedLabel
    ) {

        appendLogRow(
            data.label,
            data.confidence
        );

        lastLoggedLabel =
            data.label;
    }
}


/* ============================================================
   PROBABILITY BARS
   ============================================================ */

function renderProbBars(probabilities) {

    el.probBars.innerHTML = "";


    Object
        .entries(probabilities)
        .sort(
            (a, b) => b[1] - a[1]
        )
        .forEach(
            ([label, probability]) => {

                const row =
                    document.createElement("div");

                row.className =
                    "prob-row";


                const percentage =
                    Math.max(
                        0,
                        Math.min(
                            100,
                            probability * 100
                        )
                    );


                row.innerHTML = `

                    <span>
                        ${label}
                    </span>

                    <span class="prob-track">

                        <span
                            class="prob-fill"
                            style="
                                width:
                                ${percentage.toFixed(0)}%
                            "
                        ></span>

                    </span>

                    <span>
                        ${percentage.toFixed(0)}%
                    </span>

                `;


                el.probBars.appendChild(row);
            }
        );
}


/* ============================================================
   SESSION LOG
   ============================================================ */

function appendLogRow(
    label,
    confidence
) {

    const row =
        document.createElement("tr");


    const time =
        new Date().toLocaleTimeString(
            "en-GB",
            {
                hour12: false
            }
        );


    row.innerHTML = `

        <td>
            ${time}
        </td>

        <td>
            ${label}
        </td>

        <td>
            ${(
                confidence * 100
            ).toFixed(0)}%
        </td>

    `;


    el.logBody.prepend(row);


    while (
        el.logBody.rows.length > 30
    ) {

        el.logBody.deleteRow(-1);
    }
}


/* ============================================================
   MAIN POLLING LOOP
   ============================================================ */

/* ============================================================
   ENVIRONMENT (DHT11 + LDR + electrode status from the ESP32)
   ============================================================ */

function formatReading(value, decimals, unit) {

    if (value === null || value === undefined) {
        return "—";
    }

    return `${Number(value).toFixed(decimals)}${unit}`;
}


async function pollEnvironment() {

    const data =
        await apiGet("/api/device/status");

    const t =
        (data && data.online && data.telemetry) || {};

    el.valTemp.textContent =
        formatReading(t.temp_c, 1, " °C");

    el.valHumidity.textContent =
        formatReading(t.humidity_pct, 0, " %");

    el.valLight.textContent =
        formatReading(t.light_pct, 0, " %");

    /* lead_off = true means the AD8232 sees no electrode contact */
    if (!data || !data.online) {

        el.valElectrodes.textContent = "—";
        el.valElectrodes.classList.remove("warn");

    } else {

        el.valElectrodes.textContent =
            t.lead_off ? "Not attached" : "Attached";

        el.valElectrodes.classList.toggle(
            "warn",
            Boolean(t.lead_off)
        );
    }
}


/* ============================================================
   SLOW DC CHANNEL (ADS1115)
   ============================================================ */

const dc = {
    badge: document.getElementById("dcBadge"),
    value: document.getElementById("dcValue"),
    note: document.getElementById("dcNote"),
    chart: new Chart(
        document.getElementById("dcChart"),
        {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "DC (mV)",
                    data: [],
                    borderColor: CHART_GREEN,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.2
                }]
            },
            options: {
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: {
                    x: {
                        grid: { color: GRID_COLOR },
                        ticks: { color: TICK_COLOR, maxTicksLimit: 7 },
                        title: { display: true, text: "seconds ago", color: TICK_COLOR }
                    },
                    y: {
                        grid: { color: GRID_COLOR },
                        ticks: { color: TICK_COLOR },
                        title: { display: true, text: "mV", color: TICK_COLOR }
                    }
                }
            }
        }
    )
};


function setDcBadge(state, text) {
    dc.badge.dataset.state = state;
    dc.badge.textContent = text;
}


async function pollDc() {

    const data = await apiGet("/api/dc");
    const points = data.points || [];

    dc.chart.data.labels = points.map((p) => p[0]);
    dc.chart.data.datasets[0].data = points.map((p) => p[1]);
    dc.chart.update("none");

    if (!data.online) {
        setDcBadge("idle", "ESP32 offline");
        dc.value.textContent = "\u2014";
        dc.note.textContent = "Waiting for the ESP32.";
        dc.note.classList.remove("warn");
        return;
    }

    if (!data.ads1115) {
        setDcBadge("idle", "Waiting for ADS1115");
        dc.value.textContent = "\u2014";
        dc.note.textContent =
            "No ADS1115 found. Connect it (SDA\u2192D21, SCL\u2192D22, VDD\u21923V3, GND, ADDR\u2192GND) and restart the ESP32.";
        dc.note.classList.remove("warn");
        return;
    }

    dc.value.textContent =
        data.latest_mv === null ? "\u2014" : `${data.latest_mv.toFixed(3)} mV`;

    if (data.saturated) {
        setDcBadge("warn", "ADS1115: out of range");
        dc.note.textContent =
            "Reading is at the \u00b1256 mV limit: electrodes not attached to A0/A1, or the plant needs a reference electrode.";
        dc.note.classList.add("warn");
    } else {
        setDcBadge("live", "ADS1115: live");
        dc.note.textContent = "1-second averages, last 5 minutes.";
        dc.note.classList.remove("warn");
    }
}


/* ============================================================
   RECORDING (labelled sessions for training the model)
   ============================================================ */

const rec = {
    badge: document.getElementById("recBadge"),
    label: document.getElementById("recLabel"),
    note: document.getElementById("recNote"),
    startBtn: document.getElementById("recStartBtn"),
    stopBtn: document.getElementById("recStopBtn"),
    live: document.getElementById("recLive"),
    msg: document.getElementById("recMsg"),
    sessionsBody: document.getElementById("recSessionsBody"),
    wasRecording: null,     // so we only refresh the table when something changes
    busy: false             // true while a Start/Stop request is on its way
};


function formatDuration(seconds) {

    const total = Math.max(0, Math.round(seconds || 0));
    const m = Math.floor(total / 60);
    const s = total % 60;

    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}


function showRecMessage(text, isError = false) {

    if (!text) {
        rec.msg.hidden = true;
        rec.msg.textContent = "";
        return;
    }

    rec.msg.hidden = false;
    rec.msg.textContent = text;
    rec.msg.classList.toggle("error", isError);
}


/* Called every second from pollLoop */
async function pollRecording() {

    const data =
        await apiGet("/api/record/status");

    const recording = Boolean(data.recording);
    const session = data.session;

    /* Buttons and inputs */
    rec.startBtn.hidden = recording;
    rec.stopBtn.hidden = !recording;
    rec.label.disabled = recording;
    rec.note.disabled = recording;

    if (!rec.busy) {
        rec.startBtn.disabled = !data.device_online;
        rec.startBtn.title = data.device_online
            ? ""
            : "Connect the ESP32 first";
    }

    /* Badge */
    rec.badge.dataset.state = recording ? "recording" : "idle";
    rec.badge.textContent = recording ? "● REC" : "Idle";

    /* Live line */
    if (recording && session) {

        rec.live.classList.add("active");
        rec.live.textContent =
            `Session #${session.id} · ${session.label} · ` +
            `${formatDuration(session.duration_s)} · ` +
            `${session.sample_count} samples`;

        /* Warnings that matter for data quality */
        if (!data.device_online) {
            showRecMessage("ESP32 offline: nothing is being saved right now.", true);
        } else if (session.sample_count > 0 && session.lead_off_samples === session.sample_count) {
            showRecMessage("Electrodes not attached: every sample so far is flagged lead_off.");
        } else if (session.lead_off_samples > 0) {
            showRecMessage(`${session.lead_off_samples} samples had an electrode off.`);
        } else {
            showRecMessage("");
        }

    } else {

        rec.live.classList.remove("active");
        rec.live.textContent = data.device_online
            ? "Not recording. Pick a label, add a note, then press Start."
            : "ESP32 offline. Connect it to start recording.";
    }

    /* Refresh the sessions table when recording starts/stops */
    if (rec.wasRecording !== recording) {
        rec.wasRecording = recording;
        loadSessions();
    }
}


async function loadSessions() {

    let data;

    try {
        data = await apiGet("/api/record/sessions");
    } catch (error) {
        return;
    }

    const sessions = data.sessions || [];

    if (sessions.length === 0) {
        rec.sessionsBody.innerHTML =
            `<tr><td colspan="7" class="rec-empty">No sessions yet.</td></tr>`;
        return;
    }

    rec.sessionsBody.innerHTML = sessions.map((s) => {

        const leadOffPct = s.sample_count > 0
            ? Math.round((s.lead_off_samples / s.sample_count) * 100)
            : 0;

        const actions = s.recording
            ? `<span class="rec-live-tag">recording…</span>`
            : `<a class="rec-link" href="${BACKEND_URL}/api/record/sessions/${s.id}/csv">CSV</a>` +
              `<button class="rec-delete" data-id="${s.id}">Delete</button>`;

        return `
            <tr>
                <td>${s.id}</td>
                <td>${escapeHTML(s.label)}</td>
                <td class="rec-note" title="${escapeHTML(s.note || "")}">${escapeHTML(s.note || "—")}</td>
                <td>${formatDuration(s.duration_s)}</td>
                <td>${s.sample_count}</td>
                <td class="${leadOffPct > 0 ? "warn" : ""}">${leadOffPct}%</td>
                <td class="rec-actions">${actions}</td>
            </tr>`;
    }).join("");
}


rec.startBtn.addEventListener("click", async () => {

    rec.busy = true;
    rec.startBtn.disabled = true;
    showRecMessage("");

    try {
        await apiPost("/api/record/start", {
            label: rec.label.value,
            note: rec.note.value
        });
        await pollRecording();
    } catch (error) {
        showRecMessage(error.message, true);
    } finally {
        rec.busy = false;
    }
});


rec.stopBtn.addEventListener("click", async () => {

    rec.busy = true;
    rec.stopBtn.disabled = true;

    try {
        const result = await apiPost("/api/record/stop", {});
        const s = result.session;
        rec.note.value = "";
        await pollRecording();
        showRecMessage(
            `Saved session #${s.id}: ${s.sample_count} samples in ${formatDuration(s.duration_s)}.`
        );
    } catch (error) {
        showRecMessage(error.message, true);
    } finally {
        rec.busy = false;
        rec.stopBtn.disabled = false;
    }
});


/* Delete needs two clicks (no pop-up): first click arms it, second deletes */
rec.sessionsBody.addEventListener("click", async (event) => {

    const button = event.target.closest(".rec-delete");

    if (!button) {
        return;
    }

    if (!button.classList.contains("confirm")) {
        button.classList.add("confirm");
        button.textContent = "Sure?";
        setTimeout(() => {
            button.classList.remove("confirm");
            button.textContent = "Delete";
        }, 3000);
        return;
    }

    try {
        const response = await fetch(
            `${BACKEND_URL}/api/record/sessions/${button.dataset.id}`,
            { method: "DELETE" }
        );
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(body.error || `Delete failed (${response.status})`);
        }
        showRecMessage(`Deleted session #${button.dataset.id}.`);
    } catch (error) {
        showRecMessage(error.message, true);
    }

    loadSessions();
});


async function pollLoop() {

    try {

        await Promise.all([
            pollWaveform(),
            pollMetrics(),
            pollEnvironment(),
            pollRecording(),
            pollDc()
        ]);

        setConnectionState(true);

    } catch (error) {

        console.error(
            "Polling error:",
            error
        );

        setConnectionState(false);
    }
}


function startPolling() {

    if (state.pollHandle) {

        clearInterval(
            state.pollHandle
        );
    }


    pollLoop();


    state.pollHandle =
        setInterval(
            pollLoop,
            POLL_INTERVAL_MS
        );
}


/* ============================================================
   AI — STATUS DIAGNOSIS
   ============================================================ */

el.runDiagnosisBtn.addEventListener(
    "click",
    async () => {

        el.runDiagnosisBtn.disabled =
            true;

        el.runDiagnosisBtn.textContent =
            "Analyzing…";


        el.statusBody.innerHTML = `

            <p class="status-placeholder">
                Reading current sensor data…
            </p>

        `;


        try {

            const result =
                await apiPost(
                    "/api/ai/status",
                    {}
                );


            el.statusBody.innerHTML = `

                <p class="status-text"></p>

            `;


            el.statusBody
                .querySelector(
                    ".status-text"
                )
                .textContent =
                result.status_text;


            el.statusMeta.textContent =
                `Based on: ${
                    result.based_on.label
                } (${
                    (
                        result.based_on.confidence *
                        100
                    ).toFixed(0)
                }% confidence) — generated ${
                    new Date().toLocaleTimeString(
                        "en-GB",
                        {
                            hour12: false
                        }
                    )
                }`;

        } catch (error) {

            console.error(
                "AI diagnosis error:",
                error
            );


            el.statusBody.innerHTML = `

                <p class="status-placeholder">
                    ${escapeHTML(error.message)}
                </p>

            `;

        } finally {

            el.runDiagnosisBtn.disabled =
                false;

            el.runDiagnosisBtn.textContent =
                "Analyze now";
        }
    }
);


/* ============================================================
   AI RESPONSE FORMATTING
   ============================================================ */

/*
 * Escape HTML so AI-generated text cannot
 * accidentally inject HTML into the page.
 */

function escapeHTML(value) {

    return String(value)

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );
}


/*
 * Converts simple Markdown-style AI output
 * into a professional PhytoSense layout.
 */

function formatAIResponse(content) {

    const text =
        String(content || "").trim();


    if (!text) {

        return `
            <p class="ai-empty">
                No analysis was returned.
            </p>
        `;
    }


    const lines =
        text
            .split(/\r?\n/)
            .map(
                (line) => line.trim()
            )
            .filter(
                (line) =>
                    line.length > 0
            );


    const blocks = [];
    let listItems = [];


    function flushList() {

        if (!listItems.length) {
            return;
        }


        blocks.push(`

            <ul class="ai-list">

                ${listItems.join("")}

            </ul>

        `);


        listItems = [];
    }


    function formatInline(value) {

        let html =
            escapeHTML(value);


        /* Bold */

        html =
            html.replace(
                /\*\*(.+?)\*\*/g,
                "<strong>$1</strong>"
            );


        /* Italic */

        html =
            html.replace(
                /(^|[^\*])\*([^*]+)\*(?!\*)/g,
                "$1<em>$2</em>"
            );


        /* Inline code */

        html =
            html.replace(
                /`([^`]+)`/g,
                "<code>$1</code>"
            );


        return html;
    }


    for (const line of lines) {

        /* Bullet points */

        const bulletMatch =
            line.match(
                /^[-*•]\s+(.+)$/
            );


        if (bulletMatch) {

            listItems.push(`

                <li>
                    ${formatInline(
                        bulletMatch[1]
                    )}
                </li>

            `);

            continue;
        }


        flushList();


        /* Markdown headings */

        const headingMatch =
            line.match(
                /^#{1,3}\s+(.+)$/
            );


        if (headingMatch) {

            blocks.push(`

                <h4 class="ai-section-title">

                    ${formatInline(
                        headingMatch[1]
                    )}

                </h4>

            `);

            continue;
        }


        /*
         * Bold-only headings such as:
         *
         * **Current Assessment**
         * **Sensor Interpretation**
         * **Recommendation**
         */

        const boldOnlyMatch =
            line.match(
                /^\*\*(.+?)\*\*:?$/
            );


        if (boldOnlyMatch) {

            blocks.push(`

                <h4 class="ai-section-title">

                    ${formatInline(
                        boldOnlyMatch[1]
                    )}

                </h4>

            `);

            continue;
        }


        /* Normal paragraph */

        blocks.push(`

            <p class="ai-paragraph">

                ${formatInline(line)}

            </p>

        `);
    }


    flushList();


    return `

        <div class="ai-response-card">

            <div class="ai-response-header">

                <span class="ai-response-icon">
                    ✦
                </span>

                <span>
                    PhytoSense AI
                </span>

            </div>


            <div class="ai-response-content">

                ${blocks.join("")}

            </div>


            <div class="ai-response-footer">

                AI interpretation based on
                current sensor readings

            </div>

        </div>

    `;
}


/* ============================================================
   AI RESPONSE STYLES
   ============================================================ */

function installAIResponseStyles() {

    if (
        document.getElementById(
            "phyto-ai-response-styles"
        )
    ) {
        return;
    }


    const style =
        document.createElement("style");


    style.id =
        "phyto-ai-response-styles";


    style.textContent = `

        .chat-msg.assistant
        .chat-bubble {

            max-width:
                min(760px, 92%);

            line-height:
                1.65;
        }


        .ai-response-card {

            overflow:
                hidden;

            border:
                1px solid
                rgba(
                    127,
                    184,
                    143,
                    0.20
                );

            border-radius:
                14px;

            background:
                linear-gradient(
                    145deg,
                    rgba(
                        127,
                        184,
                        143,
                        0.055
                    ),
                    rgba(
                        255,
                        255,
                        255,
                        0.018
                    )
                );

            box-shadow:
                0 10px 30px
                rgba(
                    0,
                    0,
                    0,
                    0.12
                );
        }


        .ai-response-header {

            display:
                flex;

            align-items:
                center;

            gap:
                9px;

            padding:
                11px 14px;

            border-bottom:
                1px solid
                rgba(
                    255,
                    255,
                    255,
                    0.055
                );

            color:
                #9ed2ad;

            font-size:
                11px;

            font-weight:
                600;

            letter-spacing:
                0.08em;

            text-transform:
                uppercase;
        }


        .ai-response-icon {

            display:
                inline-flex;

            align-items:
                center;

            justify-content:
                center;

            width:
                20px;

            height:
                20px;

            border-radius:
                50%;

            background:
                rgba(
                    127,
                    184,
                    143,
                    0.12
                );

            font-size:
                12px;
        }


        .ai-response-content {

            padding:
                14px 16px 12px;
        }


        .ai-section-title {

            margin:
                15px 0 7px;

            color:
                #b7ddc1;

            font-size:
                12px;

            font-weight:
                600;

            letter-spacing:
                0.04em;

            text-transform:
                uppercase;
        }


        .ai-section-title:first-child {

            margin-top:
                0;
        }


        .ai-paragraph {

            margin:
                0 0 9px;

            color:
                rgba(
                    255,
                    255,
                    255,
                    0.88
                );

            font-size:
                13px;
        }


        .ai-paragraph:last-child {

            margin-bottom:
                0;
        }


        .ai-paragraph strong {

            color:
                #e4f3e7;

            font-weight:
                600;
        }


        .ai-list {

            margin:
                5px 0 11px;

            padding-left:
                20px;

            color:
                rgba(
                    255,
                    255,
                    255,
                    0.86
                );

            font-size:
                13px;
        }


        .ai-list li {

            margin:
                5px 0;

            padding-left:
                3px;
        }


        .ai-list li::marker {

            color:
                #7fb88f;
        }


        .ai-response-content code {

            padding:
                2px 5px;

            border-radius:
                4px;

            background:
                rgba(
                    255,
                    255,
                    255,
                    0.06
                );

            color:
                #b7ddc1;

            font-family:
                "IBM Plex Mono",
                monospace;

            font-size:
                0.9em;
        }


        .ai-response-footer {

            padding:
                9px 14px;

            border-top:
                1px solid
                rgba(
                    255,
                    255,
                    255,
                    0.045
                );

            color:
                rgba(
                    255,
                    255,
                    255,
                    0.40
                );

            font-size:
                10px;

            letter-spacing:
                0.02em;
        }


        .ai-empty {

            margin:
                0;

            color:
                rgba(
                    255,
                    255,
                    255,
                    0.60
                );
        }

    `;


    document.head.appendChild(style);
}


/* ============================================================
   AI — CHAT
   ============================================================ */

function appendChatMessage(
    role,
    content
) {

    const wrapper =
        document.createElement("div");


    wrapper.className =
        `chat-msg ${role}`;


    const bubble =
        document.createElement("div");


    bubble.className =
        "chat-bubble";


    /*
     * AI messages are professionally formatted.
     *
     * The temporary "Thinking…" message stays
     * as normal text.
     */

    if (
        role === "assistant" &&
        content !== "Thinking…"
    ) {

        bubble.innerHTML =
            formatAIResponse(content);

    } else {

        bubble.textContent =
            content;
    }


    wrapper.appendChild(
        bubble
    );


    el.chatMessages.appendChild(
        wrapper
    );


    el.chatMessages.scrollTop =
        el.chatMessages.scrollHeight;


    return bubble;
}


/* Install AI response styles */

installAIResponseStyles();


/* ============================================================
   CHAT FORM
   ============================================================ */

el.chatForm.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();


        const message =
            el.chatInput.value.trim();


        if (!message) {
            return;
        }


        /* User message */

        appendChatMessage(
            "user",
            message
        );


        /* History BEFORE this message: the backend adds the new message itself */
        const previousHistory =
            state.chatHistory.slice();


        state.chatHistory.push({
            role: "user",
            content: message
        });


        el.chatInput.value = "";


        /* Thinking message */

        const thinkingBubble =
            appendChatMessage(
                "assistant",
                "Thinking…"
            );


        try {

            const result =
                await apiPost(
                    "/api/ai/chat",
                    {
                        message,

                        history:
                            previousHistory
                    }
                );


            /*
             * Replace "Thinking…" with
             * the formatted AI response.
             */

            thinkingBubble.innerHTML =
                formatAIResponse(
                    result.reply
                );


            state.chatHistory.push({
                role: "assistant",
                content: result.reply
            });


            el.chatMessages.scrollTop =
                el.chatMessages.scrollHeight;


        } catch (error) {

            console.error(
                "AI chat error:",
                error
            );


            thinkingBubble.textContent =
                `Couldn't get a response: ${
                    error.message
                }`;


            /* Remove the unanswered message so history stays clean */
            state.chatHistory.pop();
        }
    }
);


/* ============================================================
   ENTER TO SEND CHAT
   ============================================================ */

el.chatInput.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            el.chatForm.requestSubmit();
        }
    }
);


/* ============================================================
   BOOT
   ============================================================ */

console.log(
    "PhytoSense frontend starting..."
);


console.log(
    "Chart.js:",
    Chart.version
);


console.log(
    "Backend:",
    BACKEND_URL
);


startPolling();