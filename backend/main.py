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
# Calibration done on 4 Oct 2026: fork in dry air ~2890 mV, fork in water ~1500 mV.
# Lower voltage = wetter soil. Change these two numbers if you calibrate again.
SOIL_DRY_MV = float(os.getenv("SOIL_DRY_MV", "2890"))
SOIL_WET_MV = float(os.getenv("SOIL_WET_MV", "1500"))
SOIL_HISTORY = deque(maxlen=3000)   # (unix time, mV) - about 2 days at one reading per minute
soil_busy_seq = set()               # sample numbers the soil sensor may have disturbed


def soil_percent(mv):
    """0 % = as dry as air, 100 % = as wet as a glass of water."""
    if mv is None:
        return None
    pct = (SOIL_DRY_MV - mv) / (SOIL_DRY_MV - SOIL_WET_MV) * 100.0
    return round(max(0.0, min(100.0, pct)), 1)

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
BACKEND_VERSION = "11c"
print(f"[PhytoSense] Backend version {BACKEND_VERSION} (AI monitor with Start/Stop button)")


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
                           soil_percent(SOIL_HISTORY[-1][1]) if SOIL_HISTORY else None,
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
                                "saturated": latest_b is not None and abs(latest_b) > 250.0}})


@app.route('/api/soil', methods=['GET'])
def get_soil():
    """Soil moisture: newest reading + history (one point per reading)."""
    if not SOIL_HISTORY:
        return jsonify({"available": False})
    t, mv = SOIL_HISTORY[-1]
    history = [[round(ti), soil_percent(m)] for ti, m in SOIL_HISTORY]
    return jsonify({"available": True, "mv": round(mv), "moisture_pct": soil_percent(mv),
                    "age_s": round(time.time() - t), "dry_mv": SOIL_DRY_MV, "wet_mv": SOIL_WET_MV,
                    "history": history[-720:]})


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
        status.update(display="signal_problem", label="Check electrodes")
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
                                   soil_percent(SOIL_HISTORY[-1][1]) if SOIL_HISTORY else None,
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

def ai_context():
    """Everything the AI assistant is allowed to talk about - only real, current readings."""
    if not DEVICE.is_online():
        return None
    tel = DEVICE.latest_telemetry
    return {
        "now": time.time(),
        "plant": plant_state(),
        "environment": {"temp_c": tel.get("temp_c"), "humidity_pct": tel.get("humidity_pct"),
                        "light_pct": tel.get("light_pct")},
        "fast_channel": {"electrodes_attached": not tel.get("lead_off"), "features": latest_fast or None},
        "monitor_text": MONITOR.describe(),
        "soil": ({"moisture_pct": soil_percent(SOIL_HISTORY[-1][1]),
                  "age_s": round(time.time() - SOIL_HISTORY[-1][0])} if SOIL_HISTORY else None),
    }


@app.route('/api/ai/status', methods=['POST'])
def ai_status():
    ctx = ai_context()
    if ctx is None:
        return jsonify({"error": "ESP32 is offline - no live data to analyse"}), 400
    status_text = llm.generate_status(ctx)
    return jsonify({
        "status_text": status_text,
        "based_on": {"label": ctx["plant"]["label"], "events_total": ctx["plant"]["events_total"]}
    })


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    ctx = ai_context()
    if ctx is None:
        return jsonify({"error": "ESP32 is offline - no live data to talk about"}), 400
    data = request.json or {}
    reply = llm.generate_chat_reply(data.get("message", ""), data.get("history", []), ctx)
    return jsonify({"reply": reply})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)