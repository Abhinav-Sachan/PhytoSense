import os
import re
import time
from collections import deque
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from data_stream import SensorStream, IngestBuffer
from dsp_filters import BioSignalDSP
from ai_engine import SignalFeatures
from detector import LiveDetector
from monitor import PlantMonitor
from llm_service import PlantLLM
from recorder import Recorder

app = Flask(__name__)
CORS(app)  # Allows frontend on port 5500 to fetch data from port 5000

# Samples per second for each data source
SIM_RATE = 100      # built-in simulator
DEVICE_RATE = 10    # real ESP32 (firmware sends 10 averaged samples per second)

# Simulator is OFF by default so fake data can never mix with real plant data.
# To test without the ESP32, start the server with:  $env:USE_SIMULATOR="1"; python main.py
USE_SIMULATOR = os.getenv("USE_SIMULATOR", "0") == "1"

SIM_STREAM = SensorStream(sample_rate=SIM_RATE)                    # only used when USE_SIMULATOR is on
DEVICE = IngestBuffer(max_seconds=20, expected_rate=DEVICE_RATE)   # fills up once your ESP32 talks to us
DEVICE_TOKEN = os.getenv("DEVICE_TOKEN", "phytosense-dev-token")

# ---- Where recordings are saved --------------------------------------------
# Everything is kept OUTSIDE the project folder, on the D: drive:
#   D:\PhytoSense_data\phytosense.db   <- the database (all sessions)
#   D:\PhytoSense_data\csv\            <- one CSV per session, saved automatically on Stop
# Two reasons: C: is full, and VS Code Live Server reloads the dashboard every time a
# file inside the project changes (the database changes every second while recording).
# If there is no D: drive (other computer), it falls back to backend/data/.
# You can also choose a folder yourself:  $env:PHYTOSENSE_DATA_DIR="E:\somewhere"
def pick_data_dir():
    chosen = os.getenv("PHYTOSENSE_DATA_DIR")
    if chosen:
        return chosen
    if os.name == "nt" and os.path.isdir("D:\\"):
        return "D:\\PhytoSense_data"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


DATA_DIR = pick_data_dir()
CSV_DIR = os.path.join(DATA_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)
print(f"[PhytoSense] Saving recordings in: {DATA_DIR}")

RECORDER = Recorder(db_path=os.path.join(DATA_DIR, "phytosense.db"), sample_rate=DEVICE_RATE)

# ---- Soil moisture sensor (firmware v2.2, resistive fork on VP) ---------------
# Calibration redone on 6 Oct 2026 (after box assembly): fork in dry air wanders ~2000-2400 mV
# (seen for 3 days), fork in water ~740-790 mV. (4 Oct readings were air ~2890, water ~1500.)
# Lower voltage = wetter soil. Change these two numbers if you calibrate again.
SOIL_DRY_MV = float(os.getenv("SOIL_DRY_MV", "2100"))
SOIL_WET_MV = float(os.getenv("SOIL_WET_MV", "750"))
SOIL_HISTORY = deque(maxlen=3000)   # (unix time, mV) - about 2 days at one reading per minute
soil_busy_seq = set()               # sample numbers the soil sensor may have disturbed
NOISY_SIGNAL_UV = 100.0             # normal wobble above this = bad electrode contact
SOIL_MEDIAN_OF = 5                  # show the median of the last 5 soil readings
SOIL_STALE_S = 150                  # a reading older than this is not shown (sensor reads every 60 s)


def soil_percent(mv):
    """0 % = as dry as air, 100 % = as wet as a glass of water."""
    if mv is None:
        return None
    pct = (SOIL_DRY_MV - mv) / (SOIL_DRY_MV - SOIL_WET_MV) * 100.0
    return round(max(0.0, min(100.0, pct)), 1)


def current_soil():
    """Newest soil reading as (time, mV), or None when the ESP32 is offline or the reading is old.
    Old values are never shown as if they were live."""
    if not SOIL_HISTORY or not DEVICE.is_online():
        return None
    t, _ = SOIL_HISTORY[-1]
    if time.time() - t > SOIL_STALE_S:
        return None
    # The sensor's dry reading wanders by a few hundred mV from one reading to the next
    # (tested 6 Oct). The middle value (median) of the last 5 readings hides single jumps.
    # The recordings still save every raw reading.
    recent = sorted(m for _, m in list(SOIL_HISTORY)[-SOIL_MEDIAN_OF:])
    return t, recent[len(recent) // 2]


def current_soil_pct():
    cur = current_soil()
    return soil_percent(cur[1]) if cur else None


# ---- Water status (H1, from the soil sensor) ------------------------------------------
# Rule-based, calibrated on this pot (7 Oct 2026): just after full watering the soil read
# ~64 %, after 10 hours ~52 %. Thresholds below are on the same 0-100 % scale.
WATER_WELL_PCT = 55.0       # >= : well watered
WATER_OK_PCT = 40.0         # >= : OK
WATER_THIRSTY_PCT = 25.0    # <  : thirsty, water now   (between: drying, water soon)
WATER_TREND_HOURS = 6.0     # drying speed measured over the last 6 hours
WATER_JUMP_PCT = 15.0       # soil median rising this much = it was watered
_water_cache = {"t": 0.0, "value": None}


def _median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2] if vals else None


def _water_status_uncached():
    """Is H1 getting thirsty? Uses the soil sensor: level now + how fast it is drying."""
    import bisect
    pct = current_soil_pct()
    if pct is None:
        return {"available": False,
                "reason": "offline" if not DEVICE.is_online() else "no_reading"}
    now = time.time()
    pts = list(SOIL_HISTORY)
    ts = [t for t, _ in pts]
    vs = [soil_percent(m) for _, m in pts]

    # last watering: median of the 15 min after a reading is >= 15 % above the median
    # of the 10-30 min before it (checked over the last 24 h, newest first)
    def jumped(i):
        t_i = ts[i]
        a0, a1 = i, bisect.bisect_right(ts, t_i + 900)
        b0, b1 = bisect.bisect_left(ts, t_i - 1800), bisect.bisect_right(ts, t_i - 600)
        if a1 - a0 < 3 or b1 - b0 < 3:
            return False
        return _median(vs[a0:a1]) - _median(vs[b0:b1]) >= WATER_JUMP_PCT

    last_watered = None
    first = bisect.bisect_left(ts, now - 24 * 3600)
    i = len(ts) - 1
    while i >= first and not jumped(i):
        i -= 1
    if i >= first:                       # newest watering found: take the middle of that jump
        j = i
        while j - 1 >= first and jumped(j - 1):
            j -= 1
        last_watered = ts[(i + j) // 2]

    # drying speed: straight-line fit over the last hours (% per hour, negative = drying),
    # on 10-minute medians so single jumpy readings don't matter
    trend_from = now - WATER_TREND_HOURS * 3600
    if last_watered and last_watered + 900 > trend_from:
        trend_from = last_watered + 900          # only measure drying since the last watering
    start = bisect.bisect_left(ts, trend_from)
    blocks = {}
    for t, v in zip(ts[start:], vs[start:]):
        blocks.setdefault(int(t // 600), []).append(v)
    xy = [((k * 600 + 300 - now) / 3600.0, _median(v)) for k, v in sorted(blocks.items())]
    rate = None
    if len(xy) >= 4 and xy[-1][0] - xy[0][0] >= 0.5:
        mx = sum(x for x, _ in xy) / len(xy)
        my = sum(y for _, y in xy) / len(xy)
        sxx = sum((x - mx) ** 2 for x, _ in xy)
        if sxx > 0:
            rate = sum((x - mx) * (y - my) for x, y in xy) / sxx

    if pct >= WATER_WELL_PCT:
        state, label, advice = "wet", "Well watered", "No water needed."
    elif pct >= WATER_OK_PCT:
        state, label, advice = "ok", "OK", "No water needed yet."
    elif pct >= WATER_THIRSTY_PCT:
        state, label, advice = "drying", "Getting dry", "Water soon."
    else:
        state, label, advice = "thirsty", "Thirsty", "Water now."

    hours_left = None
    if rate is not None and rate < -0.3 and pct > WATER_THIRSTY_PCT:
        hours_left = round((pct - WATER_THIRSTY_PCT) / -rate, 1)

    return {"available": True, "state": state, "label": label, "advice": advice,
            "moisture_pct": pct,
            "rate_pct_per_h": None if rate is None else round(rate, 2),
            "trend_hours": round((now - max(trend_from, ts[start] if start < len(ts) else now)) / 3600.0, 1),
            "hours_to_thirsty": hours_left,
            "last_watered": last_watered,
            "last_watered_clock": time.strftime("%d %b %H:%M", time.localtime(last_watered)) if last_watered else None,
            "basis": "soil sensor (rule-based)"}


def water_status():
    """Cached for 10 s (the dashboard asks every second or two)."""
    now = time.time()
    if _water_cache["value"] is None or now - _water_cache["t"] > 10:
        _water_cache["value"] = _water_status_uncached()
        _water_cache["t"] = now
    return _water_cache["value"]


def load_soil_history_from_db(hours=48):
    """After a restart, reload the recorded soil readings so the drying trend survives."""
    try:
        import sqlite3
        conn = sqlite3.connect(RECORDER.db_path)
        rows = conn.execute("SELECT t, soil_mv FROM samples WHERE soil_mv IS NOT NULL AND t > ? "
                            "ORDER BY t", (time.time() - hours * 3600,)).fetchall()
        conn.close()
        for t, mv in rows[-SOIL_HISTORY.maxlen:]:
            SOIL_HISTORY.append((float(t), float(mv)))
        print(f"[PhytoSense] Loaded {len(rows)} recorded soil readings (last {hours} h) for the water trend")
    except Exception as e:
        print(f"[PhytoSense] Could not load soil history: {e}")

dsp = BioSignalDSP()
signal_features = SignalFeatures(sample_rate=SIM_RATE)   # fast-channel numbers (RMS, peak, ...)
llm = PlantLLM()

# Trained plant-event detector (slow ADS1115 channel). Model file: backend/models/event_detector.json
# (made by train_detector.py from the recordings of 3 Oct 2026)
DETECTOR = LiveDetector()
if DETECTOR.model is None:
    print(f"[PhytoSense] WARNING: event detector model not loaded ({DETECTOR.model_error})")
else:
    print(f"[PhytoSense] Event detector loaded (threshold {DETECTOR.model.threshold})")

latest_fast = {}   # newest fast-channel numbers, for the AI assistant

# Background plant monitor: started with the "Start monitoring" button, keeps a
# timeline for the AI assistant. Never shows pop-ups and never calls the AI on its own.
MONITOR = PlantMonitor()
_last_monitor_update = 0.0
BACKEND_VERSION = "15c"
print(f"[PhytoSense] Backend version {BACKEND_VERSION} (chat photo check + PlantVillage disease CNN)")
load_soil_history_from_db()


def get_raw_window():
    """Returns (samples, source, sample_rate).
    source is "device" (real ESP32), "simulated" (only if USE_SIMULATOR is on)
    or "offline" (ESP32 silent, no data at all)."""
    if DEVICE.is_online():
        return DEVICE.get_latest_window(), "device", DEVICE_RATE
    if USE_SIMULATOR:
        return SIM_STREAM.get_latest_window(), "simulated", SIM_RATE
    return [], "offline", DEVICE_RATE


@app.route('/api/ingest', methods=['POST'])
def ingest():
    """The ESP32 (or our fake_esp32.py test script) POSTs here."""
    token = request.headers.get("X-Device-Token")
    if token != DEVICE_TOKEN:
        return jsonify({"error": "bad or missing X-Device-Token header"}), 401

    data = request.json or {}
    samples = data.get("bio_mv", [])
    if not isinstance(samples, list) or not samples:
        return jsonify({"error": "bio_mv must be a non-empty list of numbers"}), 400

    telemetry = {
        "temp_c": data.get("temp_c"),
        "humidity_pct": data.get("humidity_pct"),
        "light_pct": data.get("light_pct"),
        "lead_off": data.get("lead_off"),
        "rssi": data.get("rssi"),
        "uptime_s": data.get("uptime_s"),
        "ads1115": data.get("ads1115"),    # True when firmware found the ADS1115
    }
    # Optional slow DC channel from the ADS1115 (not fitted yet -> usually missing)
    dc_uv = data.get("dc_uv")
    if not isinstance(dc_uv, list):
        dc_uv = None
    # Plant B (ADS1115 A2-A3) and soil sensor (firmware v2.2+)
    dc2_uv = data.get("dc2_uv")
    if not isinstance(dc2_uv, list):
        dc2_uv = None
    soil_mv = data.get("soil_mv")
    soil_seq = data.get("soil_seq")
    if not isinstance(soil_mv, (int, float)) or not isinstance(soil_seq, int):
        soil_mv, soil_seq = None, None
    telemetry["fw"] = data.get("fw")

    # Sample numbers from firmware v2.1+ (give every sample its exact time)
    seq = data.get("seq")
    if not (isinstance(seq, list) and len(seq) == len(samples)
            and all(isinstance(x, int) for x in seq)):
        seq = None

    global soil_busy_seq
    if soil_mv is not None:
        SOIL_HISTORY.append((time.time(), float(soil_mv)))
        soil_busy_seq = {soil_seq, soil_seq + 1}

    DEVICE.push(samples, telemetry, dc_uv, dc2_uv)
    if dc_uv is not None:
        # The detector skips the 0.2 s when the soil sensor was switched on
        dc_for_detector = dc_uv
        if seq is not None and soil_busy_seq:
            dc_for_detector = [None if s in soil_busy_seq else v for v, s in zip(dc_uv, seq)]
        DETECTOR.push(dc_for_detector, seq)                        # live plant-event detection
    saved = RECORDER.add_batch(samples, telemetry, dc_uv, seq,     # saves only while a recording is running
                               dc2_uv_list=dc2_uv, soil_mv=soil_mv, soil_seq=soil_seq)

    global _last_monitor_update
    if time.time() - _last_monitor_update >= 2.0:                  # update the monitor every 2 s
        _last_monitor_update = time.time()
        try:
            MONITOR.update(plant_state(),
                           current_soil_pct(),
                           telemetry)
        except Exception as e:                                     # the monitor must never break data intake
            print(f"[PhytoSense] monitor update failed: {e}")
    return jsonify({"ok": True, "received": len(samples), "recorded": saved})


@app.route('/api/device/status', methods=['GET'])
def device_status():
    return jsonify({
        "online": DEVICE.is_online(),
        "telemetry": DEVICE.latest_telemetry,
        "source": get_raw_window()[1],
    })


def one_second_points(values):
    """10-per-second values -> [seconds ago (negative), average mV] once per second."""
    rate = DEVICE_RATE
    points = []
    for start in range(0, len(values) - rate + 1, rate):
        chunk = [v for v in values[start:start + rate] if v is not None]
        if chunk:
            seconds_ago = -(len(values) - start - rate) / rate
            points.append([round(seconds_ago, 1), round(sum(chunk) / len(chunk) / 1000.0, 4)])
    return points


@app.route('/api/dc', methods=['GET'])
def get_dc():
    """Slow DC channel (ADS1115) for the dashboard chart: plant A (A0-A1) and,
    if connected, plant B (A2-A3). 1-second averages of the last 5 minutes, in millivolts."""
    online = DEVICE.is_online()
    ads = bool(online and DEVICE.latest_telemetry.get("ads1115"))
    points = one_second_points(DEVICE.get_dc_window()) if ads else []
    points_b = one_second_points(DEVICE.get_dc2_window()) if ads else []
    latest = points[-1][1] if points else None
    latest_b = points_b[-1][1] if points_b else None
    # The ADS1115 range is +/-256 mV: readings stuck at the edge mean the inputs are
    # floating (no electrodes) or the plant is outside the range (needs a reference)
    saturated = latest is not None and abs(latest) > 250.0
    return jsonify({"online": online, "ads1115": ads, "points": points,
                    "latest_mv": latest, "saturated": saturated,
                    "plant_b": {"points": points_b, "latest_mv": latest_b,
                                "saturated": latest_b is not None and abs(latest_b) > 250.0,
                                "wobble_uv": wobble_uv(DEVICE.get_dc2_window()) if ads else None}})


def wobble_uv(raw_uv, n=100):
    """How much the signal jumps around over the last 10 s (n samples at 10/s), in microvolts.
    Robust spread: 1.4826 x median distance from the median. A plant on good pads: a few uV.
    A loose pad or soil wire picks up mains hum: hundreds of uV."""
    vals = [v for v in raw_uv[-n:] if isinstance(v, (int, float))]
    if len(vals) < 20:
        return None
    vals.sort()
    med = vals[len(vals) // 2]
    dev = sorted(abs(v - med) for v in vals)
    return round(1.4826 * dev[len(dev) // 2], 1)


@app.route('/api/soil', methods=['GET'])
def get_soil():
    """Soil moisture: newest reading + history. 'available' is False when the ESP32 is
    offline or the newest reading is too old, so the dashboard shows a dash, not old data."""
    online = DEVICE.is_online()
    history = [[round(ti), soil_percent(m)] for ti, m in SOIL_HISTORY][-720:]
    cur = current_soil()
    if not cur:
        reason = "offline" if not online else ("no_reading" if not SOIL_HISTORY else "stale")
        return jsonify({"available": False, "online": online, "reason": reason, "history": history})
    t, mv = cur
    return jsonify({"available": True, "online": online, "mv": round(mv), "moisture_pct": soil_percent(mv),
                    "age_s": round(time.time() - t), "dry_mv": SOIL_DRY_MV, "wet_mv": SOIL_WET_MV,
                    "history": history})


@app.route('/api/water', methods=['GET'])
def get_water():
    """Water status card: Well watered / OK / Getting dry / Thirsty + drying speed."""
    return jsonify(water_status())


def plant_state():
    """What the live detector says, plus whether the signal can be trusted right now."""
    online = DEVICE.is_online()
    ads = bool(online and DEVICE.latest_telemetry.get("ads1115"))
    status = DETECTOR.status()
    dc = [v for v in DEVICE.get_dc_window()[-10:] if v is not None] if ads else []
    latest_mv = (sum(dc) / len(dc) / 1000.0) if dc else None
    saturated = latest_mv is not None and abs(latest_mv) > 250.0
    status.update({"online": online, "ads1115": ads, "saturated": saturated,
                   "latest_mv": None if latest_mv is None else round(latest_mv, 4)})
    # When the signal cannot be trusted, say so instead of showing a plant state
    if not online:
        status.update(display="offline", label="ESP32 offline")
    elif not ads:
        status.update(display="no_signal", label="Slow channel not connected")
    elif saturated:
        status.update(display="signal_problem", label="Check electrodes", problem="saturated")
    elif (status.get("state") in ("calm", "event", "recovering")
          and (status.get("noise_uv") or 0) > NOISY_SIGNAL_UV):
        # A plant on good electrodes wobbles by a few microvolts. A wobble of hundreds of
        # microvolts means a pad has lost contact or the soil (reference) wire is out of the soil,
        # so the input picks up mains hum. Say so instead of "Electrical disturbance".
        status.update(display="signal_problem", label="Noisy signal: check electrodes", problem="noisy")
    else:
        status["display"] = status["state"]
    return status


@app.route('/api/monitor', methods=['GET'])
def get_monitor():
    """What the background monitor knows: monitoring since when, trends, timeline."""
    global _last_monitor_update
    if time.time() - _last_monitor_update > 5.0:     # ESP32 silent: still let the monitor notice
        _last_monitor_update = time.time()
        MONITOR.update(plant_state(), None, DEVICE.latest_telemetry)
    return jsonify(dict(MONITOR.summary(), backend_version=BACKEND_VERSION))


@app.route('/api/monitor/start', methods=['POST'])
def monitor_start():
    ok, err = MONITOR.start_manual(plant_state(),
                                   current_soil_pct(),
                                   DEVICE.latest_telemetry)
    if not ok:
        return jsonify({"error": err}), 409
    return jsonify({"ok": True, "monitor": dict(MONITOR.summary(), backend_version=BACKEND_VERSION)})


@app.route('/api/monitor/stop', methods=['POST'])
def monitor_stop():
    ok, err = MONITOR.stop_manual()
    if not ok:
        return jsonify({"error": err}), 409
    return jsonify({"ok": True, "monitor": dict(MONITOR.summary(), backend_version=BACKEND_VERSION)})


@app.route('/api/state', methods=['GET'])
def get_state():
    """Live plant state for the dashboard: Calm / Stress event detected / Recovering."""
    return jsonify(plant_state())


@app.route('/api/waveform', methods=['GET'])
def get_waveform():
    raw, source, rate = get_raw_window()
    filtered = dsp.process(raw, rate) if raw else []
    return jsonify({"raw": raw, "filtered": list(filtered), "source": source, "sample_rate": rate})


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Fast-channel (AD8232) signal numbers. No fake classification any more:
    the plant state comes from /api/state (trained detector)."""
    global latest_fast
    raw, source, rate = get_raw_window()
    if len(raw) < 50:
        latest_fast = {}   # no fresh data -> the AI must not talk about old readings
        return jsonify({"ready": False, "source": source})

    filtered = dsp.process(raw, rate)
    signal_features.sample_rate = rate   # so the dominant frequency is calculated correctly
    latest_fast = signal_features.extract(filtered)
    return jsonify({"ready": True, "features": latest_fast, "source": source})


# ---- Recording (labelled sessions for training the model) -----------------

@app.route('/api/record/status', methods=['GET'])
def record_status():
    status = RECORDER.status()
    status["device_online"] = DEVICE.is_online()
    return jsonify(status)


@app.route('/api/record/start', methods=['POST'])
def record_start():
    if not DEVICE.is_online():
        return jsonify({"error": "ESP32 is offline - connect it before recording"}), 409
    data = request.json or {}
    session, error = RECORDER.start(data.get("label"), data.get("note", ""))
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"ok": True, "session": session})


def csv_filename(session):
    """e.g. phytosense_session12_mechanical_pinch-3.csv"""
    name = f"phytosense_session{session['id']}_{session['label']}"
    note = re.sub(r"[^A-Za-z0-9]+", "-", session.get("note") or "").strip("-")[:40]
    if note:
        name += "_" + note
    return name + ".csv"


def save_csv_to_disk(session_id):
    """Writes the session's CSV into CSV_DIR. Returns the full path, or None."""
    session = RECORDER.get_session(session_id)
    csv_text = RECORDER.export_csv(session_id)
    if session is None or csv_text is None:
        return None
    path = os.path.join(CSV_DIR, csv_filename(session))
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(csv_text)
    return path


# Long "drought" recordings are cut into 1-hour parts; save each finished part as CSV too
def _save_finished_part(session_id):
    path = save_csv_to_disk(session_id)
    print(f"[PhytoSense] 1-hour part finished, CSV saved: {path}")


RECORDER.on_auto_split = _save_finished_part


@app.route('/api/record/stop', methods=['POST'])
def record_stop():
    session, error = RECORDER.stop()
    if error:
        return jsonify({"error": error}), 400
    try:
        saved_path = save_csv_to_disk(session["id"])
        print(f"[PhytoSense] CSV saved: {saved_path}")
    except Exception as e:   # recording is safe in the database even if this fails
        saved_path = None
        print(f"[PhytoSense] Could not save CSV file: {e}")
    return jsonify({"ok": True, "session": session, "csv_path": saved_path})


@app.route('/api/record/sessions', methods=['GET'])
def record_sessions():
    return jsonify({"sessions": RECORDER.list_sessions()})


@app.route('/api/record/sessions/<int:session_id>/csv', methods=['GET'])
def record_export(session_id):
    csv_text = RECORDER.export_csv(session_id)
    if csv_text is None:
        return jsonify({"error": "session not found"}), 404
    session = RECORDER.get_session(session_id)
    filename = csv_filename(session)
    return Response(csv_text, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route('/api/record/sessions/<int:session_id>', methods=['DELETE'])
def record_delete(session_id):
    ok, error = RECORDER.delete_session(session_id)
    if not ok:
        return jsonify({"error": error}), 400
    return jsonify({"ok": True})


# ---- AI assistant ----------------------------------------------------------

LAST_LEAF_CHECK = None      # {"t": time, "result": {...}} - the AI assistant can talk about it


def ai_context():
    """Everything the AI assistant is allowed to talk about - only real, current readings.
    Works offline too (then it only knows that the ESP32 is offline + the last leaf check)."""
    now = time.time()
    if not DEVICE.is_online():
        return {"now": now, "online": False, "leaf_check": LAST_LEAF_CHECK}
    tel = DEVICE.latest_telemetry
    raw_b = DEVICE.get_dc2_window()
    b_vals = [v for v in raw_b[-10:] if isinstance(v, (int, float))]
    cur_soil = current_soil()
    return {
        "now": now,
        "online": True,
        "plant": plant_state(),
        "plant_b": {"latest_mv": round(sum(b_vals) / len(b_vals) / 1000.0, 4) if b_vals else None,
                    "wobble_uv": wobble_uv(raw_b)},
        "environment": {"temp_c": tel.get("temp_c"), "humidity_pct": tel.get("humidity_pct"),
                        "light_pct": tel.get("light_pct")},
        "fast_channel": {"electrodes_attached": not tel.get("lead_off"), "features": latest_fast or None},
        "monitor_text": MONITOR.describe(),
        "soil": ({"moisture_pct": soil_percent(cur_soil[1]), "age_s": round(now - cur_soil[0])}
                 if cur_soil else None),
        "water": water_status(),
        "leaf_check": LAST_LEAF_CHECK,
    }


@app.route('/api/ai/status', methods=['POST'])
def ai_status():
    ctx = ai_context()
    if not ctx.get("online"):
        return jsonify({"error": "ESP32 is offline - no live data to analyse"}), 400
    try:
        status_text = llm.generate_status(ctx)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({
        "status_text": status_text,
        "based_on": {"label": ctx["plant"]["label"], "events_total": ctx["plant"]["events_total"]}
    })


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """Chat with the assistant. If a photo is attached (the "+" button), the AI vision model
    identifies the plant, checks its health and answers the question about the photo."""
    global LAST_LEAF_CHECK
    data = request.json or {}
    message = (data.get("message") or "").strip()
    image = data.get("image") or ""
    if image:
        if not image.startswith("data:image/"):
            return jsonify({"error": "Please attach a photo (JPG or PNG)"}), 400
        if len(image) > MAX_LEAF_IMAGE_CHARS:
            return jsonify({"error": "Photo is too large - please use a smaller photo"}), 400
        question = message or "What plant is this, and is it healthy?"
        ctx = ai_context()
        cnn = None
        if LEAF_CNN and data.get("pixels"):
            try:
                cnn = LEAF_CNN.predict_b64(data["pixels"])
            except Exception as e:
                print(f"[PhytoSense] Leaf CNN failed: {e}")
        cnn_info = (f"38 classes, {LEAF_CNN.test_accuracy:.0%} accuracy on its test photos"
                    if LEAF_CNN and LEAF_CNN.test_accuracy else None)
        try:
            result = llm.check_leaf(image, None, ctx if ctx.get("online") else None, question=question,
                                    cnn=cnn, cnn_info=cnn_info)
        except Exception as e:
            return jsonify({"error": str(e)}), 502
        result["cnn"] = cnn
        result["cnn_accuracy"] = LEAF_CNN.test_accuracy if LEAF_CNN else None
        result["sensor_link"] = leaf_sensor_link(result, ctx)
        LAST_LEAF_CHECK = {"t": time.time(), "note": question, "result": result}
        reply = result.get("answer") or result.get("summary") or "I looked at the photo - see the result below."
        return jsonify({"reply": reply, "leaf": dict(result, checked_at=time.strftime("%d %b %H:%M"))})
    if not message:
        return jsonify({"error": "Empty message"}), 400
    try:
        reply = llm.generate_chat_reply(message, data.get("history", []), ai_context())
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"reply": reply})


from leaf_model import load_classifier
LEAF_CNN = load_classifier()        # PlantVillage disease CNN (NumPy only); None if the file is missing

MAX_LEAF_IMAGE_CHARS = 5_500_000     # ~4 MB image after base64 (Groq's limit for inline images)


@app.route('/api/leaf/check', methods=['POST'])
def leaf_check():
    """Leaf Health Check: a photo (data URL, resized in the browser) -> AI vision model ->
    verdict, visible symptoms, possible causes, advice. Combined with live sensor readings."""
    global LAST_LEAF_CHECK
    data = request.json or {}
    image = data.get("image") or ""
    if not image.startswith("data:image/"):
        return jsonify({"error": "Please choose a photo (JPG or PNG)"}), 400
    if len(image) > MAX_LEAF_IMAGE_CHARS:
        return jsonify({"error": "Photo is too large - please use a smaller photo"}), 400
    note = (data.get("note") or "").strip()[:200]
    ctx = ai_context()
    try:
        result = llm.check_leaf(image, note, ctx if ctx.get("online") else None)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    result["sensor_link"] = leaf_sensor_link(result, ctx)
    LAST_LEAF_CHECK = {"t": time.time(), "note": note, "result": result}
    return jsonify(dict(result, checked_at=time.strftime("%d %b %H:%M")))


def leaf_sensor_link(result, ctx):
    """One sentence that connects what the photo shows with what the sensors measure."""
    if not ctx.get("online"):
        return "Sensors offline: this result is based on the photo only."
    water = ctx.get("water") or {}
    causes = " ".join((c.get("cause") or "").lower() for c in (result.get("possible_causes") or []))
    if not water.get("available"):
        return "No soil reading right now: this result is based on the photo only."
    pct = water["moisture_pct"]
    if "underwater" in causes or "drought" in causes or "dry" in causes:
        if pct < 40:
            return f"The soil sensor agrees: soil moisture is only {pct:.0f} % ({water['label']})."
        return f"The soil sensor shows {pct:.0f} % moisture, so lack of water is less likely right now."
    if "overwater" in causes or "root rot" in causes:
        if pct > 70:
            return f"The soil sensor agrees: the soil is very wet ({pct:.0f} %)."
        return f"The soil sensor shows {pct:.0f} % moisture, so overwatering is less likely right now."
    return f"Soil moisture is {pct:.0f} % ({water['label']}); water does not look like the main issue."


@app.route('/api/leaf/last', methods=['GET'])
def leaf_last():
    if not LAST_LEAF_CHECK:
        return jsonify({"available": False})
    return jsonify(dict(LAST_LEAF_CHECK["result"], available=True,
                        checked_at=time.strftime("%d %b %H:%M", time.localtime(LAST_LEAF_CHECK["t"]))))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)