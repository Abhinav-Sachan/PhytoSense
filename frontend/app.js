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
 *               → Plant state (trained event detector, /api/state)
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

    plantState:
        document.getElementById("plantState"),

    stateLabel:
        document.getElementById("stateLabel"),

    stateSub:
        document.getElementById("stateSub"),

    stateMeter:
        document.getElementById("stateMeter"),

    stateMark:
        document.getElementById("stateMark"),

    stateMeta:
        document.getElementById("stateMeta"),

    logNote:
        document.getElementById("logNote"),

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

    valSoil:
        document.getElementById("valSoil"),

    valSoilSub:
        document.getElementById("valSoilSub"),

    valPlantB:
        document.getElementById("valPlantB"),

    valPlantBSub:
        document.getElementById("valPlantBSub"),

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
   FAST CHANNEL NUMBERS (AD8232)
   ============================================================ */

async function pollMetrics() {

    const data =
        await apiGet("/api/metrics");

    if (!data) {
        return;
    }

    if (!data.ready || state.leadOff) {
        /* Don't show numbers when no data is coming in or the AD8232 pads are not attached */
        el.valRms.textContent = "—";
        el.valPeak.textContent = "—";
        el.valFreq.textContent = "—";
        el.valZcr.textContent = "—";
        return;
    }

    const features = data.features || {};

    el.valRms.textContent =
        Number(features.rms || 0).toFixed(3);

    el.valPeak.textContent =
        Number(features.max_amplitude || 0).toFixed(3);

    el.valFreq.textContent =
        `${Number(features.dominant_frequency || 0).toFixed(1)} Hz`;

    el.valZcr.textContent =
        Number(features.zero_crossing_rate || 0).toFixed(3);
}


/* ============================================================
   PLANT STATE (trained event detector on the slow ADS1115 channel)
   ============================================================ */

function formatClock(unixSeconds) {

    return new Date(unixSeconds * 1000).toLocaleTimeString(
        "en-GB",
        { hour12: false }
    );
}


function formatSeconds(seconds) {

    if (seconds === null || seconds === undefined) {
        return "—";
    }

    const s = Math.round(seconds);

    return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}


async function pollState() {

    const data =
        await apiGet("/api/state");

    if (!data) {
        return;
    }

    const display = data.display || data.state;

    el.plantState.dataset.state = display;
    el.stateLabel.textContent = data.label || "—";


    /* One line explaining the current state */

    let sub = "";

    if (display === "offline") {
        sub = "Waiting for ESP32…";
    } else if (display === "no_signal") {
        sub = "The ADS1115 is not sending data. Check its wires and restart the ESP32.";
    } else if (display === "signal_problem" && data.problem === "noisy") {
        sub = `The signal is wobbling by ±${Math.round(data.noise_uv || 0)} µV (a plant on good electrodes wobbles by a few µV). ` +
              "A pad may have lost contact (re-gel it) or the soil wire is out of the soil.";
    } else if (display === "signal_problem") {
        sub = "The voltage is at the edge of the range — an electrode or the soil wire may be loose.";
    } else if (display === "no_model") {
        sub = "models/event_detector.json not found — run train_detector.py.";
    } else if (display === "warming_up") {
        sub = `Learning this plant's normal signal… ${data.warmup_left_s} s left`;
    } else if (display === "event" && data.current_event) {
        sub = `Something is disturbing the plant · ${formatSeconds(data.current_event.duration_s)} so far · ` +
              `change ${(data.current_event.peak_mv * 1000).toFixed(0)} µV`;
    } else if (display === "recovering") {
        sub = `Signal is settling after the event · back to Calm in ${data.recover_left_s} s`;
    } else if (display === "calm") {
        sub = "No disturbance detected. The plant's signal is at its normal level.";
    }

    el.stateSub.textContent = sub;


    /* Probability meter with the alarm threshold marked */

    const live = ["calm", "event", "recovering"].includes(display);
    const probability = live ? (data.probability || 0) : 0;

    el.stateMeter.style.width = `${(probability * 100).toFixed(0)}%`;

    el.stateMark.style.left =
        `${((data.threshold || 0.9) * 100).toFixed(0)}%`;

    el.stateMeta.textContent = live
        ? `Event probability ${(probability * 100).toFixed(0)}% · ` +
          `normal wobble ±${Math.round(data.noise_uv || 0)} µV · ` +
          `now ${data.deviation_uv >= 0 ? "+" : ""}${Math.round(data.deviation_uv || 0)} µV from normal`
        : "";


    renderEventLog(data);
}


function renderEventLog(data) {

    const rows = [];

    if (data.current_event) {
        rows.push({ ...data.current_event, live: true });
    }

    (data.recent_events || []).forEach((e) => rows.push(e));

    if (rows.length === 0) {
        el.logBody.innerHTML =
            '<tr><td colspan="5" class="rec-empty">No events yet.</td></tr>';
    } else {
        el.logBody.innerHTML = rows.map((e) => `
            <tr class="${e.live ? "event-live" : ""}">
                <td>${formatClock(e.started)}</td>
                <td>${e.live ? "Happening now" : `Event #${e.id}`}</td>
                <td>${(e.peak_mv * 1000).toFixed(0)} µV</td>
                <td>${formatSeconds(e.duration_s)}</td>
                <td>${Math.round((e.confidence || 0) * 100)}%</td>
            </tr>
        `).join("");
    }


    /* How good the detector is (from its honest test) */

    const ev = data.model && data.model.evaluation;
    const trained = data.model && data.model.trained_on;

    el.logNote.textContent = (ev && trained)
        ? `Detector trained on ${trained.sessions} recordings (${trained.date}). ` +
          `In testing it found ${ev.events_detected} of ${ev.events_total} touch/pinch events, ` +
          `with about ${ev.false_alarms_per_hour} false alarms per hour.`
        : "";
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
    state.leadOff = Boolean(data && data.online && t.lead_off);
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
   LIVE MONITORING (AI Assistant page) - updates by itself, no AI call
   ============================================================ */

const mon = {
    box: document.getElementById("monitorBox"),
    title: document.getElementById("monitorTitle"),
    facts: document.getElementById("monitorFacts"),
    timeline: document.getElementById("monitorTimeline"),
    btn: document.getElementById("monitorBtn"),
    on: false
};


mon.btn.addEventListener("click", async () => {
    const stopping = mon.on;                         // what the button shows right now
    mon.btn.disabled = true;
    mon.btn.textContent = stopping ? "Stopping\u2026" : "Starting\u2026";
    try {
        const result = await apiPost(stopping ? "/api/monitor/stop" : "/api/monitor/start", {});
        renderMonitor(result.monitor);               // show the new state immediately
    } catch (error) {
        const old = /404|405/.test(error.message);
        mon.facts.textContent = old
            ? "The server is still running the old version. Stop Flask (Ctrl+C) and start it again: python main.py"
            : error.message;
        mon.errorUntil = Date.now() + 8000;          // keep the message visible for a few seconds
        mon.btn.textContent = stopping ? "Stop Monitoring" : "Start Monitoring";
    } finally {
        mon.btn.disabled = false;
        mon.btn.textContent = mon.on ? "Stop Monitoring" : "Start Monitoring";
    }
});


function trendText(t, unit, decimals) {
    if (!t) return null;
    const a = Number(t.from).toFixed(decimals), b = Number(t.to).toFixed(decimals);
    return a === b ? `steady at ${b}${unit}` : `${a}${unit} \u2192 ${b}${unit} (last ${t.minutes} min)`;
}


async function pollMonitor() {
    if (mon.btn.disabled) return;                    // a Start/Stop click is in progress
    const data = await apiGet("/api/monitor");
    if (data) renderMonitor(data);
}


function renderMonitor(data) {
    if (!data) return;

    mon.box.dataset.on = data.monitoring ? "true" : "false";
    mon.on = !!data.monitoring;
    if (!mon.btn.disabled) {
        mon.btn.textContent = data.monitoring ? "Stop Monitoring" : "Start Monitoring";
    }

    if (!data.backend_version && !(mon.errorUntil && Date.now() < mon.errorUntil)) {
        mon.facts.textContent = "The server is still running the old version. Stop Flask (Ctrl+C) and start it again: python main.py";
        return;
    }

    if (data.monitoring) {
        mon.title.textContent = `Monitoring since ${data.since_clock} \u00b7 ${data.duration}`;
        const facts = [`${data.disturbances} disturbance${data.disturbances === 1 ? "" : "s"}`];
        const soil = trendText(data.trends.soil, " %", 0);
        const temp = trendText(data.trends.temp, " \u00b0C", 1);
        if (soil) facts.push(`soil ${soil}`);
        if (temp) facts.push(`temperature ${temp}`);
        mon.facts.textContent = facts.join(" \u00b7 ");
    } else if (!(mon.errorUntil && Date.now() < mon.errorUntil)) {
        mon.title.textContent = "Not monitoring";
        mon.facts.textContent = "Put the electrodes on the plant, then press Start Monitoring.";
    }

    mon.timeline.innerHTML = (data.timeline || []).slice(0, 8).map((e) =>
        `<li data-kind="${escapeHTML(e.kind)}"><span>${escapeHTML(e.clock)}</span>${escapeHTML(e.text)}</li>`
    ).join("");
}


/* ============================================================
   SOIL MOISTURE (resistive sensor, firmware v2.2)
   ============================================================ */

/* ============================================================
   WATER STATUS (H1, rule-based from the soil sensor)
   ============================================================ */

const water = {
    box: document.getElementById("waterState"),
    label: document.getElementById("waterLabel"),
    sub: document.getElementById("waterSub"),
    meta: document.getElementById("waterMeta")
};

async function pollWater() {
    const d = await apiGet("/api/water");
    if (!d || !d.available) {
        water.box.dataset.state = "offline";
        water.label.textContent = "—";
        water.sub.textContent = !d ? "Backend offline" :
            d.reason === "offline" ? "ESP32 offline" : "Waiting for a soil reading…";
        water.meta.textContent = "";
        return;
    }
    water.box.dataset.state = d.state;
    water.label.textContent = d.label;

    let sub = `Soil moisture ${Math.round(d.moisture_pct)} % · ${d.advice}`;
    if (d.hours_to_thirsty !== null && d.hours_to_thirsty !== undefined) {
        sub += ` Thirsty in about ${d.hours_to_thirsty < 1 ? "1" : Math.round(d.hours_to_thirsty)} h at this speed.`;
    }
    water.sub.textContent = sub;

    const parts = [];
    if (d.rate_pct_per_h !== null && d.rate_pct_per_h !== undefined) {
        parts.push(`trend ${d.rate_pct_per_h >= 0 ? "+" : ""}${d.rate_pct_per_h.toFixed(1)} %/h (last ${d.trend_hours} h)`);
    } else {
        parts.push("trend: needs 30 min of readings");
    }
    if (d.last_watered_clock) parts.push(`last watered ${d.last_watered_clock}`);
    water.meta.textContent = parts.join(" · ");
}


async function pollSoil() {

    const data = await apiGet("/api/soil");

    /* Offline or old reading: show a dash, never an old number as if it were live */
    if (!data || !data.available) {
        el.valSoil.textContent = "—";
        el.valSoil.classList.remove("warn");
        el.valSoilSub.textContent =
            !data ? "" :
            data.reason === "offline" ? "ESP32 offline" :
            data.reason === "stale" ? "waiting for a new reading" :
            "no reading yet";
        return;
    }

    el.valSoil.textContent = `${Math.round(data.moisture_pct)} %`;
    el.valSoil.classList.toggle("warn", data.moisture_pct < 20);

    const age = data.age_s < 90 ? `${data.age_s} s ago` : `${Math.round(data.age_s / 60)} min ago`;
    el.valSoilSub.textContent = `${data.mv} mV · ${age}`;
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

    /* Plant B (A2-A3): newest value + how much it wobbles (pad contact check).
       Nothing on A2-A3 reads as a flat 0.000 mV, so a flat zero means "not connected". */
    const b = data.plant_b || {};
    const bVals = (b.points || []).slice(-10).map((p) => p[1]).filter((v) => v !== null && v !== undefined);
    const bFlatZero = bVals.length >= 3 && bVals.every((v) => Math.abs(v) < 0.005);
    const bWobble = (b.wobble_uv === null || b.wobble_uv === undefined) ? null : b.wobble_uv;
    el.valPlantB.classList.remove("warn");
    el.valPlantBSub.textContent = "";
    if (!data.online || b.latest_mv === null || b.latest_mv === undefined) {
        el.valPlantB.textContent = "—";
    } else if (b.saturated || bFlatZero) {
        el.valPlantB.textContent = "Not connected";
    } else if (bWobble !== null && bWobble > 100) {
        el.valPlantB.textContent = "Noisy signal";
        el.valPlantB.classList.add("warn");
        el.valPlantBSub.textContent = `wobble ±${Math.round(bWobble)} µV · re-press H2 pads, push soil wire deeper`;
    } else {
        el.valPlantB.textContent = `${b.latest_mv.toFixed(3)} mV`;
        if (bWobble !== null) el.valPlantBSub.textContent = `wobble ±${bWobble.toFixed(1)} µV · OK`;
    }

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
            pollState(),
            pollSoil(),
            pollWater(),
            pollMonitor(),
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
                    result.based_on.events_total
                } events detected so far) — generated ${
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

/* Photo attached with the "+" button (data URL, already shrunk) */
const chatPhoto = {
    plus: document.getElementById("chatPlus"),
    file: document.getElementById("chatFile"),
    box: document.getElementById("chatAttach"),
    img: document.getElementById("chatAttachImg"),
    remove: document.getElementById("chatAttachRemove"),
    data: null
};

/* Centre square of the photo shrunk to 96 x 96, as raw RGB bytes in base64 -
   the input of the PlantVillage disease CNN on the server */
function squarePixels(dataUrl, size = 96) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onerror = () => reject(new Error("Could not read the photo"));
        img.onload = () => {
            const side = Math.min(img.width, img.height);
            const canvas = document.createElement("canvas");
            canvas.width = size; canvas.height = size;
            const g = canvas.getContext("2d");
            g.imageSmoothingQuality = "high";
            g.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, size, size);
            const rgba = g.getImageData(0, 0, size, size).data;
            let bin = "";
            for (let i = 0; i < rgba.length; i += 4) {
                bin += String.fromCharCode(rgba[i], rgba[i + 1], rgba[i + 2]);
            }
            resolve(btoa(bin));
        };
        img.src = dataUrl;
    });
}

function clearChatPhoto() {
    chatPhoto.pixels = null;
    chatPhoto.data = null;
    chatPhoto.file.value = "";
    chatPhoto.box.hidden = true;
}

chatPhoto.plus.addEventListener("click", () => chatPhoto.file.click());
chatPhoto.remove.addEventListener("click", clearChatPhoto);
chatPhoto.file.addEventListener("change", async () => {
    const f = chatPhoto.file.files && chatPhoto.file.files[0];
    if (!f) return;
    try {
        chatPhoto.data = await shrinkImage(f);
        const originalUrl = URL.createObjectURL(f);          /* original photo: sharper input for the CNN */
        try { chatPhoto.pixels = await squarePixels(originalUrl); } finally { URL.revokeObjectURL(originalUrl); }
        chatPhoto.img.src = chatPhoto.data;
        chatPhoto.box.hidden = false;
        el.chatInput.focus();
    } catch (e) {
        appendChatMessage("assistant", `Couldn't use that file: ${e.message}`);
        clearChatPhoto();
    }
});

/* What our own PlantVillage disease CNN said */
function cnnLineHTML(r) {
    if (!r.cnn || !r.cnn.length) return "";
    const best = r.cnn[0];
    const acc = r.cnn_accuracy ? ` · ${Math.round(r.cnn_accuracy * 100)}% test accuracy` : "";
    if (r.cnn_used === false) {
        return `<div class="leaf-meta" style="margin-top:6px">Disease model (PlantVillage CNN): this plant is not one of its 14 crops, so it was not used.</div>`;
    }
    return `<div class="cnn-line"><strong>Disease model (PlantVillage CNN${acc}):</strong>
        ${escapeHTML(best.crop)} – ${escapeHTML(best.condition)} (${Math.round(best.probability * 100)}%)</div>`;
}

/* Small result card under the assistant's answer to a photo */
function leafCardHTML(r) {
    const items = (arr) => (arr && arr.length) ? `<ul>${arr.map((x) => `<li>${escapeHTML(x)}</li>`).join("")}</ul>` : "";
    const causes = (r.possible_causes || []).map((c) => `${c.cause || "?"}${c.likelihood ? ` (${c.likelihood})` : ""}`);
    return `
        <div class="chat-leaf-card">
            <span class="leaf-verdict" data-level="${verdictLevel(r.verdict)}">${escapeHTML(r.verdict || "Unclear")}</span>
            <div class="leaf-meta">${escapeHTML(r.plant_guess || "plant not recognised")} · confidence ${escapeHTML(r.confidence || "?")}</div>
            ${r.symptoms && r.symptoms.length ? `<strong>Visible signs</strong>${items(r.symptoms)}` : ""}
            ${causes.length ? `<strong>Possible causes</strong>${items(causes)}` : ""}
            ${r.sensor_link ? `<div class="leaf-link">${escapeHTML(r.sensor_link)}</div>` : ""}
            ${cnnLineHTML(r)}
            <div class="leaf-meta" style="margin-top:6px">AI visual check, not a lab diagnosis</div>
        </div>`;
}

el.chatForm.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        const message = el.chatInput.value.trim();
        const image = chatPhoto.data;
        const pixels = chatPhoto.pixels;

        if (!message && !image) {
            return;
        }

        /* User message (with the photo, if any) */
        const userBubble = appendChatMessage("user", message || "What plant is this, and is it healthy?");
        if (image) {
            const photo = document.createElement("img");
            photo.className = "chat-photo";
            photo.src = image;
            photo.alt = "Plant photo";
            userBubble.prepend(photo);
        }

        /* History BEFORE this message: the backend adds the new message itself */
        const previousHistory = state.chatHistory.slice();
        const historyText = (message || "What plant is this, and is it healthy?") + (image ? " [plant photo attached]" : "");
        state.chatHistory.push({ role: "user", content: historyText });

        el.chatInput.value = "";
        clearChatPhoto();

        const thinkingBubble = appendChatMessage("assistant", image ? "Looking at the photo…" : "Thinking…");

        try {
            const payload = { message, history: previousHistory };
            if (image) { payload.image = image; payload.pixels = pixels; }

            const result = await apiPost("/api/ai/chat", payload);

            thinkingBubble.innerHTML = formatAIResponse(result.reply) + (result.leaf ? leafCardHTML(result.leaf) : "");

            state.chatHistory.push({ role: "assistant", content: result.reply });

            el.chatMessages.scrollTop = el.chatMessages.scrollHeight;

        } catch (error) {
            console.error("AI chat error:", error);
            thinkingBubble.textContent = `Couldn't get a response: ${error.message}`;
            state.chatHistory.pop();   /* keep history clean */
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

/* ============================================================
   CHAT SUGGESTION CHIPS
   ============================================================ */

document.querySelectorAll("#chatChips .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
        el.chatInput.value = chip.dataset.q;
        el.chatForm.requestSubmit();
    });
});


/* ============================================================
   LEAF HEALTH CHECK (photo -> AI vision model)
   ============================================================ */

/* Shrink the photo in the browser (max 1024 px, JPEG) so upload is fast and small */
function shrinkImage(file, maxSide = 1024) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(new Error("Could not read the photo"));
        reader.onload = () => {
            const img = new Image();
            img.onerror = () => reject(new Error("This file is not a photo"));
            img.onload = () => {
                const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
                const canvas = document.createElement("canvas");
                canvas.width = Math.round(img.width * scale);
                canvas.height = Math.round(img.height * scale);
                canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
                resolve(canvas.toDataURL("image/jpeg", 0.85));
            };
            img.src = reader.result;
        };
        reader.readAsDataURL(file);
    });
}

function verdictLevel(v) {
    const t = String(v || "").toLowerCase();
    if (t.includes("healthy")) return "healthy";
    if (t.includes("mild")) return "mild";
    if (t.includes("moderate")) return "moderate";
    if (t.includes("severe")) return "severe";
    return "unknown";
}

