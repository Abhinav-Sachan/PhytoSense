"""
PhytoSense - plant monitor (the AI assistant's memory)
=======================================================

Runs quietly in the background. It never shows pop-ups and never calls the AI by itself.
It only keeps an up-to-date picture of what is happening, so that when you ask the
assistant a question it already knows the whole story.

What it does
------------
1. Monitoring is started with the "Start monitoring" button (you confirm the electrodes
   are on the plant). It stops with "Stop monitoring", or by itself if the ESP32 goes
   offline or the signal hits the range limit.
   Why not fully automatic: tested 4 Oct 2026 - pads lying on the table give a quiet
   signal that looks exactly like a calm plant. The ADS1115 has no electrode-contact
   check, so software alone cannot tell "on a leaf" from "on the table".
2. While monitoring, writes a short timeline of notable things:
     - electrical disturbances (from the live detector) and when they ended
     - soil moisture changes (5 % or more)
     - temperature changes (2 C or more) and light switched on / off
3. Keeps one reading per minute (soil, temperature, humidity, light, plant signal level)
   to describe trends over the last hour.

"Real plant signal" (rule of thumb, checked on 3-4 Oct 2026 recordings):
   ESP32 online + ADS1115 found + voltage inside the +/-250 mV range +
   normal wobble below 100 microvolts.
   (Empty inputs wandered with a wobble of about 120-150 microvolts or hit the range limit.)
"""

import time
from collections import deque

CONNECT_AFTER_S = 30        # signal must look like a plant this long before monitoring starts
DISCONNECT_AFTER_S = 30     # and must look wrong this long before monitoring stops
MAX_PLANT_WOBBLE_UV = 100.0
SOIL_STEP_PCT = 5.0
TEMP_STEP_C = 2.0
LIGHT_STEP_PCT = 30.0
MINUTE = 60.0


def _fmt_duration(seconds):
    seconds = int(max(0, seconds))
    h, m = seconds // 3600, (seconds % 3600) // 60
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min"
    return f"{seconds} s"


def _clock(t):
    return time.strftime("%H:%M", time.localtime(t))


class PlantMonitor:
    def __init__(self):
        self.monitoring = False
        self.since = None
        self._ok_since = None
        self._bad_since = None
        self.timeline = deque(maxlen=60)       # newest last: {"t", "kind", "text"}
        self.minutes = deque(maxlen=24 * 60)   # one snapshot per minute while monitoring
        self._last_minute = 0.0
        self._seen_events = 0
        self._in_event = False
        self._ref_soil = None
        self._ref_temp = None
        self._ref_light = None
        self.disturbances = 0

    # ---- helpers -------------------------------------------------------------
    def _log(self, now, kind, text):
        self.timeline.append({"t": now, "kind": kind, "text": text})

    @staticmethod
    def signal_looks_like_plant(plant):
        """plant = the dict from plant_state() in main.py"""
        if not plant.get("online") or not plant.get("ads1115") or plant.get("saturated"):
            return False
        if plant.get("state") in ("warming_up", "no_model"):
            return None                         # can't judge yet
        noise = plant.get("noise_uv")
        return noise is not None and noise < MAX_PLANT_WOBBLE_UV

    # ---- called regularly (every ingest, throttled in main.py) -----------------
    # ---- buttons ------------------------------------------------------------------
    def start_manual(self, plant, soil_pct=None, env=None, now=None):
        now = time.time() if now is None else now
        if self.monitoring:
            return False, "already monitoring"
        if not plant.get("online"):
            return False, "ESP32 is offline"
        if not plant.get("ads1115"):
            return False, "ADS1115 is not sending data"
        self._start(now, plant, soil_pct, env or {})
        return True, None

    def stop_manual(self, now=None):
        now = time.time() if now is None else now
        if not self.monitoring:
            return False, "not monitoring"
        self._log(now, "stop", f"Monitoring stopped by user after {_fmt_duration(now - self.since)}")
        self.monitoring = False
        self.since = None
        return True, None

    def update(self, plant, soil_pct=None, env=None, now=None):
        now = time.time() if now is None else now
        env = env or {}

        # 1. stop by itself only when the signal is clearly gone (offline / out of range)
        lost = (not plant.get("online")) or (not plant.get("ads1115")) or plant.get("saturated")
        if lost:
            self._bad_since = self._bad_since or now
            if self.monitoring and now - self._bad_since >= DISCONNECT_AFTER_S:
                self._stop(now, plant)
        else:
            self._bad_since = None

        if not self.monitoring:
            return

        # 2. disturbances from the live detector
        total = plant.get("events_total") or 0
        if total > self._seen_events:
            cur = plant.get("current_event") or {}
            peak = cur.get("peak_mv")
            extra = f" (change {peak * 1000:+.0f} µV)" if peak is not None else ""
            self._log(now, "disturbance", f"Electrical disturbance detected{extra}")
            self.disturbances += total - self._seen_events
            self._seen_events = total
            self._in_event = True
        if self._in_event and plant.get("state") != "event":
            recent = (plant.get("recent_events") or [{}])[0]
            dur = recent.get("duration_s")
            self._log(now, "recovered", "Disturbance ended" + (f" after {dur:.0f} s" if dur else ""))
            self._in_event = False

        # 3. soil, temperature, light changes
        if soil_pct is not None:
            if self._ref_soil is None:
                self._ref_soil = soil_pct
            elif abs(soil_pct - self._ref_soil) >= SOIL_STEP_PCT:
                word = "rose" if soil_pct > self._ref_soil else "fell"
                self._log(now, "soil", f"Soil moisture {word} from {self._ref_soil:.0f} % to {soil_pct:.0f} %")
                self._ref_soil = soil_pct
        temp = env.get("temp_c")
        if temp is not None:
            if self._ref_temp is None:
                self._ref_temp = temp
            elif abs(temp - self._ref_temp) >= TEMP_STEP_C:
                self._log(now, "temperature", f"Temperature changed from {self._ref_temp:.1f} °C to {temp:.1f} °C")
                self._ref_temp = temp
        light = env.get("light_pct")
        if light is not None:
            if self._ref_light is None:
                self._ref_light = light
            elif abs(light - self._ref_light) >= LIGHT_STEP_PCT:
                word = "brighter" if light > self._ref_light else "darker"
                self._log(now, "light", f"Room got {word} (light {self._ref_light:.0f} % -> {light:.0f} %)")
                self._ref_light = light

        # 4. one snapshot per minute for trends
        if now - self._last_minute >= MINUTE:
            self._last_minute = now
            self.minutes.append({"t": now, "soil": soil_pct, "temp": temp,
                                 "humidity": env.get("humidity_pct"), "light": light,
                                 "baseline_uv": plant.get("baseline_uv"), "noise_uv": plant.get("noise_uv")})

    def _start(self, now, plant, soil_pct, env):
        self.monitoring = True
        self.since = now
        self._seen_events = plant.get("events_total") or 0
        self._in_event = False
        self.disturbances = 0
        self._ref_soil, self._ref_temp, self._ref_light = soil_pct, env.get("temp_c"), env.get("light_pct")
        self.minutes.clear()
        self._last_minute = 0.0
        self._log(now, "start", "Monitoring started (electrodes on the plant, confirmed by user)")

    def _stop(self, now, plant):
        why = "ESP32 offline" if not plant.get("online") else "signal out of range (electrode or soil wire loose?)"
        self._log(now, "stop", f"Monitoring stopped: {why}, after {_fmt_duration(now - self.since)}")
        self.monitoring = False
        self.since = None

    # ---- output ----------------------------------------------------------------
    def _trend(self, key, now, window_s=3600):
        pts = [m for m in self.minutes if m.get(key) is not None and now - m["t"] <= window_s]
        if len(pts) < 2:
            return None
        return {"from": pts[0][key], "to": pts[-1][key], "minutes": round((pts[-1]["t"] - pts[0]["t"]) / 60)}

    def summary(self, now=None):
        now = time.time() if now is None else now
        out = {
            "monitoring": self.monitoring,
            "since": self.since,
            "since_clock": _clock(self.since) if self.since else None,
            "duration": _fmt_duration(now - self.since) if self.since else None,
            "disturbances": self.disturbances if self.monitoring else 0,
            "waiting_s": None,
            "trends": {k: self._trend(k, now) for k in ("soil", "temp", "humidity", "light", "baseline_uv")},
            "timeline": [dict(e, clock=_clock(e["t"])) for e in reversed(self.timeline)][:15],
        }
        return out

    def describe(self, now=None):
        """Plain-text story for the AI assistant."""
        now = time.time() if now is None else now
        s = self.summary(now)
        lines = []
        if s["monitoring"]:
            lines.append(f"Monitoring this plant since {s['since_clock']} ({s['duration']}); "
                         f"{s['disturbances']} electrical disturbance(s) so far.")
            t = s["trends"]
            if t["soil"]:
                lines.append(f"Soil moisture over the last {t['soil']['minutes']} min: "
                             f"{t['soil']['from']:.0f} % -> {t['soil']['to']:.0f} %")
            if t["temp"]:
                lines.append(f"Temperature over the last {t['temp']['minutes']} min: "
                             f"{t['temp']['from']:.1f} C -> {t['temp']['to']:.1f} C")
            if t["baseline_uv"]:
                lines.append(f"Plant signal level over the last {t['baseline_uv']['minutes']} min: "
                             f"{t['baseline_uv']['from']:+.0f} uV -> {t['baseline_uv']['to']:+.0f} uV")
        else:
            lines.append("Not monitoring right now (the user has not pressed Start monitoring, "
                         "or the ESP32 went offline).")
        if s["timeline"]:
            lines.append("Recent timeline (newest first):")
            for e in s["timeline"][:10]:
                lines.append(f"  {e['clock']}  {e['text']}")
        return "\n".join(lines)
