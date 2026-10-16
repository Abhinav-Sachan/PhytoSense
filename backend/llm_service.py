import os
import re

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# What the assistant must know about how PhytoSense works and what it can NOT do.
# Keep this honest: it is shown to visitors at the exhibition.
SYSTEM_FACTS = """You are PhytoSense, the friendly assistant of a low-cost plant monitoring system
built by B.Tech students. You talk to students, teachers and exhibition visitors.

HOW THE SYSTEM WORKS
- Gel electrode pads on the leaves of money plants, a reference pad on the stem.
- An ADS1115 16-bit converter reads the slow leaf voltage of TWO plants (H1 on A0-A1, H2 on A2-A3)
  with about 1 microvolt resolution, 10 times per second. An ESP32 sends it over Wi-Fi to a Python
  Flask server that stores everything (SQLite + CSV) and runs the analysis.
- A soil moisture sensor sits in H1's pot (0 % = dry air, 100 % = water). Also temperature,
  humidity (DHT11) and light (LDR).
- A live detector (logistic regression on 9 features, tested with leave-one-session-out: found
  10 of 12 events, about 0.8 false alarms per hour) flags sudden electrical disturbances.
- A rule-based Water status model (from the soil sensor) says Well watered / OK / Getting dry /
  Thirsty, measures the drying speed and predicts when water will be needed.
- Photo check (the "+" button in the chat): the user sends a plant photo; an AI vision model
  identifies the plant and describes visible symptoms and likely causes. For 14 crops it is helped
  by the team's own disease CNN trained on the PlantVillage dataset (38 classes: healthy + common
  diseases of apple, corn, grape, potato, tomato, etc.). It is a visual AI check, not a lab
  diagnosis.

CURRENT EXPERIMENT (drought, started 7 Oct 2026)
- H1 gets no water, H2 is watered daily as the control. Both recorded day and night.
- Day 1-2: H1 soil fell from about 62 % to about 48 %. No clear drought signal in the leaf voltage
  yet; a published 16-plant tomato study saw water stress in the electrical signal after ~4 days.

HONEST FINDINGS (share them when relevant, they are real science)
- Finger-touch dips came mainly from the human body's electricity: a plastic pen gave no dip and
  an AD8232 amplifier with interference cancelling saw no dip. So the detector shows that the
  plant was disturbed, not the plant's own stress response.
- Room lights and mains wiring ADD electrical noise. Switching the room lights OFF made the
  signal much cleaner: noise fell from 10-24 microvolts (lights on) to about 1 microvolt (lights off).
- A plastic pen touch gave NO signal; a finger touch did (human body electricity).

WHAT YOU CAN AND CANNOT SAY
- Disease: only from the Leaf Health Check photo result (visual signs). The electrical signal
  can NOT diagnose disease; that is future research (a published tomato powdery-mildew
  electrophysiology dataset exists).
- Water: use the Water status and soil numbers below.
- Never invent readings: use only the numbers given below. If something is missing, say so.

PHOTOS
- Photos are analysed only when the user attaches one with the "+" button next to the chat box.
  If the user asks about a photo but there is no "Latest Leaf Health Check" result in the readings
  below, say: "Please press + and attach the photo again so I can look at it." Do not guess.
- If a photo result exists, answer questions about the plant from that result (causes, care,
  precautions) and do not talk about the drought experiment unless asked.

HOW TO ANSWER
- Start with the direct answer in one sentence, then 2-4 short bullet points if useful.
- Simple words, no jargon unless asked. Keep it under about 150 words unless asked for more.
- If asked about the project, explain it proudly but honestly.
"""


def _fmt_ago(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h {(seconds % 3600) // 60} min ago"


def describe(ctx):
    """Turns the live context into plain text lines for the model."""
    now = ctx["now"]
    lines = []
    leaf = ctx.get("leaf_check")
    if not ctx.get("online"):
        lines.append("The ESP32 is OFFLINE right now: there are no live sensor readings. "
                     "Answer general questions and explain this if asked about live values.")
        if leaf:
            lines.append(_describe_leaf(leaf, now))
        return "\n".join(lines)
    p = ctx["plant"]
    ev = (p.get("model") or {}).get("evaluation")
    if ev:
        lines.append(f"Detector test result: found {ev['events_detected']} of {ev['events_total']} "
                     f"touch events (mostly human-body electricity), about "
                     f"{ev['false_alarms_per_hour']} false alarms per hour of calm signal")
    lines.append(f"Current plant state: {p['label']}")
    if p.get("display") in ("calm", "event", "recovering"):
        lines.append(f"Event probability right now: {p['probability']:.0%} "
                     f"(alarm threshold {p['threshold']:.0%})")
        if p.get("noise_uv") is not None:
            lines.append(f"Normal signal wobble: about {p['noise_uv']:.0f} microvolts; "
                         f"current distance from normal: {p['deviation_uv']:+.0f} microvolts")
    if p.get("display") == "warming_up":
        lines.append(f"The detector needs {p['warmup_left_s']} more seconds of signal before it can judge.")
    if p.get("current_event"):
        e = p["current_event"]
        lines.append(f"Event in progress for {e['duration_s']:.0f} s, biggest change so far {e['peak_mv']:+.3f} mV.")
    if p.get("recover_left_s") is not None:
        lines.append(f"Recovering: {p['recover_left_s']} s left before returning to Calm.")
    lines.append(f"Events detected since the server started: {p['events_total']}")
    for e in p.get("recent_events", [])[:5]:
        lines.append(f"  - event #{e['id']}: {_fmt_ago(now - e['started'])}, lasted {e['duration_s']:.0f} s, "
                     f"biggest change {e['peak_mv']:+.3f} mV, detector confidence {e['confidence']:.0%}")
    pb = ctx.get("plant_b") or {}
    if pb.get("latest_mv") is not None:
        lines.append(f"Plant B / H2 (A2-A3): {pb['latest_mv']:+.3f} mV"
                     + (f", wobble about {pb['wobble_uv']:.0f} microvolts" if pb.get("wobble_uv") is not None else ""))
    soil = ctx.get("soil")
    if soil and soil.get("moisture_pct") is not None:
        lines.append(f"Soil moisture: {soil['moisture_pct']:.0f} % (measured {_fmt_ago(soil['age_s'])})")
    water = ctx.get("water") or {}
    if water.get("available"):
        w = f"Water status (rule-based, from the soil sensor): {water['label']} - {water['advice']}"
        if water.get("rate_pct_per_h") is not None:
            w += f" Soil moisture changing {water['rate_pct_per_h']:+.1f} % per hour over the last {water.get('trend_hours')} hours."
        if water.get("hours_to_thirsty") is not None:
            w += f" At this speed it reaches 'Thirsty' in about {water['hours_to_thirsty']:.0f} hours."
        if water.get("last_watered_clock"):
            w += f" Last watering detected: {water['last_watered_clock']}."
        lines.append(w)
    env = ctx.get("environment", {})
    parts = []
    if env.get("temp_c") is not None:
        parts.append(f"temperature {env['temp_c']:.1f} C")
    if env.get("humidity_pct") is not None:
        parts.append(f"humidity {env['humidity_pct']:.0f} %")
    if env.get("light_pct") is not None:
        parts.append(f"light {env['light_pct']:.0f} %")
    if parts:
        lines.append("Room: " + ", ".join(parts))
    if leaf:
        lines.append(_describe_leaf(leaf, now))
    if ctx.get("monitor_text"):
        lines.append("")
        lines.append(ctx["monitor_text"])
    return "\n".join(lines)


def _describe_leaf(leaf, now):
    r = leaf.get("result") or {}
    txt = (f"Latest Leaf Health Check ({_fmt_ago(now - leaf['t'])}): verdict '{r.get('verdict', '?')}', "
           f"plant looks like: {r.get('plant_guess', '?')}. Symptoms: {', '.join(r.get('symptoms') or []) or 'none seen'}. ")
    causes = r.get("possible_causes") or []
    if causes:
        txt += "Possible causes: " + "; ".join(f"{c.get('cause')} ({c.get('likelihood')})" for c in causes) + ". "
    if r.get("advice"):
        adv = r["advice"]
        txt += "Advice given: " + (" ".join(adv) if isinstance(adv, list) else str(adv))
    return txt


TEXT_MODELS = ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
# Groq's image-capable models (Oct 2026 docs: qwen/qwen3.8-27b). Older names kept as fallbacks;
# if none of them exist, every other chat model on the key is tried once (see _vision_candidates).
VISION_MODEL_PREFERENCE = ["qwen/qwen3.8-27b",
                           "meta-llama/llama-4-scout-17b-16e-instruct",
                           "meta-llama/llama-4-maverick-17b-128e-instruct"]
NOT_CHAT_MODELS = ("whisper", "tts", "guard", "playai", "distil", "orpheus", "embed")

LEAF_PROMPT = """You are a plant health assistant. Look carefully at this leaf / plant photo.
Return ONLY a JSON object (no other text) with these keys:
{
  "is_plant": true or false,
  "plant_guess": "most likely plant: common name (scientific name), e.g. money plant (Epipremnum aureum), tomato (Solanum lycopersicum), or unknown",
  "plant_facts": "1-2 sentences about this plant: how to care for it (water, light)",
  "verdict": one of "Healthy", "Mild stress", "Moderate stress", "Severe stress", "Not a plant",
  "symptoms": ["short visible signs, e.g. yellowing at edges, brown dry tips, dark spots with yellow halo, white powder, holes, wilting"],
  "possible_causes": [{"cause": "e.g. underwatering, overwatering, fungal leaf spot, powdery mildew, nutrient (nitrogen) deficiency, sunburn, pests", "likelihood": "high | medium | low"}],
  "advice": ["2-4 short practical steps"],
  "confidence": "high | medium | low",
  "summary": "one plain sentence for a student"
}
Rules: only describe what is visible; if the photo is blurry or not a plant say so; do not
claim a lab-confirmed diagnosis."""


def _parse_json(text):
    """Pull the first {...} block out of a model reply and parse it."""
    import json
    import re
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)     # reasoning models (e.g. Qwen)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class PlantLLM:

    def __init__(self):
        # Initialize Groq using the API key from .env
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Please check the backend/.env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = TEXT_MODELS[0]
        self.vision_model = os.getenv("LEAF_VISION_MODEL") or None   # found automatically if empty

    # ---- text ---------------------------------------------------------------------------
    def _complete(self, messages, max_tokens, temperature):
        """Try the main text model, fall back to the next one if Groq refuses it."""
        last_error = None
        for model in [self.model] + [m for m in TEXT_MODELS if m != self.model]:
            try:
                res = self.client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
                text = (res.choices[0].message.content or "")
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                if text:
                    self.model = model
                    return text
            except Exception as e:      # model removed / rate limit: try the next one
                last_error = e
                print(f"[PhytoSense] AI model {model} failed: {e}")
        raise RuntimeError(f"AI service not available right now ({last_error})")

    def generate_status(self, ctx):
        prompt = (
            "Live readings:\n" + describe(ctx) + "\n\n"
            "Write a short status report with these headings: Overall, Signal, Water, "
            "Environment. One or two short sentences each. If a Leaf Health Check exists, "
            "add a heading Leaf. End with one practical recommendation."
        )
        return self._complete([{"role": "system", "content": SYSTEM_FACTS},
                               {"role": "user", "content": prompt}], 700, 0.3)

    def generate_chat_reply(self, message, history, ctx):
        messages = [{"role": "system", "content": SYSTEM_FACTS + "\nLive readings:\n" + describe(ctx)}]
        for msg in (history or [])[-12:]:          # last 6 questions + answers is enough context
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": str(msg["content"])[:2000]})
        messages.append({"role": "user", "content": str(message)[:2000]})
        return self._complete(messages, 900, 0.4)

    # ---- leaf photo -----------------------------------------------------------------------
    def _vision_candidates(self):
        """Models to try for a photo, best first. The working one is remembered."""
        if self.vision_model:
            yield self.vision_model
        try:
            ids = [m.id for m in self.client.models.list().data]
            print(f"[PhytoSense] Groq models on this key: {', '.join(sorted(ids))}")
        except Exception as e:
            print(f"[PhytoSense] Could not list Groq models: {e}")
            ids = []
        tried = {self.vision_model}
        likely = [i for i in ids if any(k in i.lower() for k in ("vision", "-vl", "llama-4", "scout", "maverick", "qwen"))]
        others = [i for i in ids if i not in likely and not any(k in i.lower() for k in NOT_CHAT_MODELS)]
        order = [m for m in VISION_MODEL_PREFERENCE if not ids or m in ids] + likely + others
        for m in order[:10]:
            if m not in tried:
                tried.add(m)
                yield m

    def check_leaf(self, image_data_url, note, ctx, question=None, cnn=None, cnn_info=None):
        """Leaf/plant photo -> JSON verdict. With a chat question, the JSON also gets an
        "answer" key: a direct, friendly reply to that question."""
        sensor_text = describe(ctx) if ctx else "No live sensor data."
        ask = ""
        if question:
            ask = ('\nThe user asked in the chat: "' + question[:500] + '"\n'
                   'Also add the key "answer": a direct, friendly reply to that question in 2-5 short '
                   'sentences (you may use **bold** and "- " bullet lines), based on the photo and the '
                   'sensor readings. Start by naming the plant.')
        if cnn:
            preds = "; ".join(f"{c['crop']} - {c['condition']} ({c['probability']:.0%})" for c in cnn)
            ask += ("\nA specialist disease model (CNN trained by the PhytoSense team on the PlantVillage "
                    "dataset" + (f", {cnn_info}" if cnn_info else "") + ") predicts for this photo: " + preds + ".\n"
                    "It ONLY knows these crops: apple, blueberry, cherry, corn, grape, orange, peach, bell pepper, "
                    "potato, raspberry, soybean, squash, strawberry, tomato, and was trained on photos of single "
                    "leaves. If the plant in the photo is one of these crops, give its prediction strong weight "
                    "and mention it. If the plant is anything else (for example a money plant), ignore it. "
                    'Add the key "cnn_used": true if you relied on it, otherwise false.')
        user_content = [
            {"type": "text", "text": LEAF_PROMPT + ask + (f"\nUser note: {note}" if note and not question else "")
                + "\nSensor readings for this plant (use them only to rank water-related causes):\n"
                + sensor_text[:1500]},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        last_error = None
        for model in self._vision_candidates():
            try:
                res = self.client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": user_content}],
                    max_tokens=900, temperature=0.2)
                text = res.choices[0].message.content or ""
                data = _parse_json(text)
                if data is None:
                    data = {"verdict": "Unclear", "summary": text.strip()[:500], "symptoms": [],
                            "possible_causes": [], "advice": [], "confidence": "low"}
                self.vision_model = model
                data["model"] = model
                return data
            except Exception as e:
                last_error = e
                print(f"[PhytoSense] Vision model {model} failed: {e}")
        raise RuntimeError("No AI vision model accepted the photo"
                           + (f" (last error: {last_error})" if last_error else " (no models found on the Groq key)"))
