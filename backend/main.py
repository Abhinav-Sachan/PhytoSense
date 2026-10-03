import os
import re
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from data_stream import SensorStream, IngestBuffer
from dsp_filters import BioSignalDSP
from ai_engine import PlantClassifier
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

dsp = BioSignalDSP()
classifier = PlantClassifier(sample_rate=SIM_RATE)
llm = PlantLLM()

latest_metrics = {}


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

    # Sample numbers from firmware v2.1+ (give every sample its exact time)
    seq = data.get("seq")
    if not (isinstance(seq, list) and len(seq) == len(samples)
            and all(isinstance(x, int) for x in seq)):
        seq = None

    DEVICE.push(samples, telemetry, dc_uv)
    saved = RECORDER.add_batch(samples, telemetry, dc_uv, seq)   # saves only while a recording is running
    return jsonify({"ok": True, "received": len(samples), "recorded": saved})


@app.route('/api/device/status', methods=['GET'])
def device_status():
    return jsonify({
        "online": DEVICE.is_online(),
        "telemetry": DEVICE.latest_telemetry,
        "source": get_raw_window()[1],
    })


@app.route('/api/dc', methods=['GET'])
def get_dc():
    """Slow DC channel (ADS1115, electrodes on A0/A1) for the dashboard chart.
    Returns 1-second averages of the last 5 minutes, in millivolts."""
    online = DEVICE.is_online()
    ads = bool(online and DEVICE.latest_telemetry.get("ads1115"))
    values = DEVICE.get_dc_window() if ads else []
    rate = DEVICE_RATE
    points = []   # [seconds ago (negative), average mV]
    for start in range(0, len(values) - rate + 1, rate):
        chunk = [v for v in values[start:start + rate] if v is not None]
        if chunk:
            seconds_ago = -(len(values) - start - rate) / rate
            points.append([round(seconds_ago, 1), round(sum(chunk) / len(chunk) / 1000.0, 4)])
    latest = points[-1][1] if points else None
    # The ADS1115 range is +/-256 mV: readings stuck at the edge mean the inputs are
    # floating (no electrodes) or the plant is outside the range (needs a reference)
    saturated = latest is not None and abs(latest) > 250.0
    return jsonify({"online": online, "ads1115": ads, "points": points,
                    "latest_mv": latest, "saturated": saturated})


@app.route('/api/waveform', methods=['GET'])
def get_waveform():
    raw, source, rate = get_raw_window()
    filtered = dsp.process(raw, rate) if raw else []
    return jsonify({"raw": raw, "filtered": list(filtered), "source": source, "sample_rate": rate})


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    global latest_metrics
    raw, source, rate = get_raw_window()
    if len(raw) < 50:
        latest_metrics = {}   # no fresh data -> the AI must not talk about old readings
        return jsonify({"ready": False, "source": source})

    filtered = dsp.process(raw, rate)
    classifier.sample_rate = rate   # so the dominant frequency is calculated correctly
    features = classifier.extract_features(filtered)
    classification = classifier.classify(features)

    latest_metrics = {
        "ready": True, "label": classification["label"],
        "confidence": classification["confidence"],
        "probabilities": classification["probabilities"], "features": features,
        "source": source,
    }
    return jsonify(latest_metrics)


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

@app.route('/api/ai/status', methods=['POST'])
def ai_status():
    if not latest_metrics:
        return jsonify({"error": "No data yet"}), 400
    status_text = llm.generate_status(latest_metrics)
    return jsonify({
        "status_text": status_text,
        "based_on": {"label": latest_metrics["label"], "confidence": latest_metrics["confidence"]}
    })


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    if not latest_metrics:
        return jsonify({"error": "No data yet"}), 400
    data = request.json
    reply = llm.generate_chat_reply(data.get("message", ""), data.get("history", []), latest_metrics)
    return jsonify({"reply": reply})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)