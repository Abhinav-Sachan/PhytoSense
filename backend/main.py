from flask import Flask, jsonify, request
from flask_cors import CORS
from data_stream import SensorStream
from dsp_filters import BioSignalDSP
from ai_engine import PlantClassifier
from llm_service import PlantLLM

app = Flask(__name__)
CORS(app) # Allows frontend on port 5500 to fetch data from port 5000

stream = SensorStream(sample_rate=100)
dsp = BioSignalDSP(sample_rate=100)
classifier = PlantClassifier(sample_rate=100)
llm = PlantLLM()

latest_metrics = {}

@app.route('/api/waveform', methods=['GET'])
def get_waveform():
    raw = stream.get_latest_window()
    filtered = dsp.process(raw) if raw else []
    return jsonify({"raw": raw, "filtered": list(filtered)})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    global latest_metrics
    raw = stream.get_latest_window()
    if len(raw) < 50:
        return jsonify({"ready": False})
        
    filtered = dsp.process(raw)
    features = classifier.extract_features(filtered)
    classification = classifier.classify(features)
    
    latest_metrics = {
        "ready": True, "label": classification["label"],
        "confidence": classification["confidence"],
        "probabilities": classification["probabilities"], "features": features
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
    data = request.json
    reply = llm.generate_chat_reply(data.get("message", ""), data.get("history", []), latest_metrics)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)