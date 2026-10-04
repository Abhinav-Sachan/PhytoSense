import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# What the assistant must know about how PhytoSense works and what it can NOT do.
# Keep this honest: it is shown to visitors at the exhibition.
SYSTEM_FACTS = """You are PhytoSense, the assistant of a plant electrophysiology monitor.

How the system works:
- Two electrode pads on two leaves of a money plant, a reference wire in the soil.
- An ADS1115 16-bit converter reads the slow voltage between the two leaves (microvolts),
  10 times per second, sent by an ESP32 over Wi-Fi. A second plant can be read on A2-A3.
- A resistive soil moisture sensor (switched on only 0.1 s per minute) gives soil moisture
  in percent: 0 % = as dry as air, 100 % = as wet as water.
- A detector (logistic regression on 9 signal features, trained on 18 labelled 5-minute
  recordings from 3 Oct 2026) watches for sudden electrical disturbances. States: "Calm",
  "Electrical disturbance detected", "Recovering" (the first minute after a disturbance).

What the team found out (be open about this, it is a real scientific result):
- Touching the plant with a finger produced clear dips, and the detector learned them.
- Control tests then showed these dips come mainly from the electricity of the human body:
  touching with a plastic pen gave no dip, and an AD8232 amplifier with built-in
  interference cancelling also saw no dip from a finger touch.
- So the detector reliably detects that someone touched or disturbed the plant, but it does
  NOT measure the plant's own stress response. A money plant's own signals are slow
  (minutes to days); that is why the next experiment tracks drought over several days,
  using the soil moisture sensor as the reference.

You also receive a monitoring timeline that the system keeps by itself (when monitoring
started, disturbances, soil/temperature/light changes, trends). Use it to answer questions
about what happened over time, e.g. "what happened in the last hour?".

Limits you must respect:
- It can NOT diagnose disease, nutrients or water stress yet; the drought model is being built.
- Never invent readings. Only use the numbers given below. Never call a result a
  scientifically proven diagnosis. Use simple language for students and visitors.
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
    p = ctx["plant"]
    now = ctx["now"]
    lines = []
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
    soil = ctx.get("soil")
    if soil and soil.get("moisture_pct") is not None:
        lines.append(f"Soil moisture: {soil['moisture_pct']:.0f} % (measured {_fmt_ago(soil['age_s'])})")
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
    if ctx.get("monitor_text"):
        lines.append("")
        lines.append(ctx["monitor_text"])
    return "\n".join(lines)


class PlantLLM:

    def __init__(self):
        # Initialize Groq using the API key from .env
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Please check the backend/.env file."
            )

        self.client = Groq(api_key=api_key)

        # Use the current Groq model directly.
        # This prevents an old LLM_MODEL value in .env
        # from selecting a decommissioned model.
        self.model = "openai/gpt-oss-20b"

    def generate_status(self, ctx):
        prompt = (
            SYSTEM_FACTS
            + "\nLive readings:\n" + describe(ctx) + "\n\n"
            "Write a short plain-language status (2-3 sentences) of what the plant's "
            "electrical signal is doing right now and what recent events mean. "
            "Mention a limit only if it matters for this reading."
        )
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3
        )
        return res.choices[0].message.content

    def generate_chat_reply(self, message, history, ctx):
        system_context = SYSTEM_FACTS + "\nLive readings:\n" + describe(ctx)

        formatted_history = [{"role": "system", "content": system_context}]

        # Add previous conversation messages
        for msg in history:
            if "role" in msg and "content" in msg:
                formatted_history.append({"role": msg["role"], "content": msg["content"]})

        # Add the current user message
        formatted_history.append({"role": "user", "content": message})

        res = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_history,
            max_tokens=400,
            temperature=0.5
        )
        return res.choices[0].message.content
