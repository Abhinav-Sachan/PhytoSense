import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from data_stream import SensorStream, IngestBuffer
from dsp_filters import BioSignalDSP
from ai_engine import PlantClassifier
from llm_service import PlantLLM

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
    }
    DEVICE.push(samples, telemetry)
    return jsonify({"ok": True, "received": len(samples)})


@app.route('/api/device/status', methods=['GET'])
def device_status():
    return jsonify({
        "online": DEVICE.is_online(),
        "telemetry": DEVICE.latest_telemetry,
        "source": get_raw_window()[1],
    })


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