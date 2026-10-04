"""
PhytoSense - live plant event detector
======================================

What it does (in simple words)
------------------------------
The slow channel (ADS1115, electrodes on two leaves) gives one number every 0.1 s
(dc_uv, in microvolts). This file finds sudden changes ("electrical disturbances") in
that signal, live: a short dip of about 0.1 - 0.3 mV for 3 - 8 seconds.

Honest note (tested 3 Oct 2026): the touch/pinch events it was trained on turned out to
come mostly from the electricity of the human body touching the plant (a plastic-pen
touch gave no event; the AD8232 with interference cancelling saw no event). So this is
a DISTURBANCE detector (something/someone touched or disturbed the plant), not a
detector of the plant's own stress response.

How it works
------------
1. Every 5 samples (0.5 s) we take their average  -> a "block".
2. Baseline = median of the blocks from 30 s ago to 4 s ago (the plant's "normal" level).
   Noise    = how much the plant normally wobbles around that baseline (robust spread).
3. We look at the last 4 s (8 blocks) and compute a few numbers (the FEATURES), mostly
   "how far below/above normal is the signal, measured in units of normal wobble".
4. A small trained model (logistic regression, trained by train_detector.py on the
   recordings of 3 Oct 2026) turns the features into a probability of "event".
5. A state machine turns probabilities into what the dashboard shows:
       Calm  ->  Stress event detected  ->  Recovering  ->  Calm

The SAME feature code is used for training and for the live dashboard, so they always match.
"""

import json
import math
import os
import time
from collections import deque

SAMPLE_RATE = 10            # dc_uv samples per second (firmware v2.1)
BLOCK = 5                   # samples per block -> 0.5 s blocks
BLOCK_S = BLOCK / SAMPLE_RATE
WINDOW_BLOCKS = 8           # last 4 s are inspected
GAP_BLOCKS = 8              # baseline stops 4 s before "now" (so the event itself is not in it)
BASE_BLOCKS = 60            # baseline uses 30 s of blocks
NOISE_BLOCKS = 120          # noise uses 60 s of blocks
HISTORY_BLOCKS = NOISE_BLOCKS + GAP_BLOCKS     # 64 s of data needed before we can judge
NOISE_FLOOR_UV = 0.2        # smallest "normal wobble" allowed (was 10 µV, which hid small dips on a quiet plant)

FEATURE_NAMES = [
    "mean_z",        # average distance from normal over the last 4 s, in units of wobble
    "min_z",         # deepest point (dips are negative)
    "max_z",         # highest point
    "std_z",         # how jumpy the last 4 s were, compared with normal wobble
    "range_z",       # max - min
    "frac_out",      # fraction of the last 8 blocks that were more than 3 wobbles away
    "slope_z",       # change from the first to the last block of the window
    "mean_uv_100",   # average distance from normal, in units of 100 microvolts
    "noise_uv_100",  # the normal wobble itself, in units of 100 microvolts
]

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "event_detector.json")


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def features_from_blocks(blocks):
    """blocks: list of 0.5-s block averages (microvolts), oldest first, at least
    HISTORY_BLOCKS long. Uses the newest HISTORY_BLOCKS. Returns (feature_list, info)."""
    b = list(blocks)[-HISTORY_BLOCKS:]
    window = b[-WINDOW_BLOCKS:]
    history = b[:-GAP_BLOCKS]                       # 60 s that end 4 s before now
    baseline = _median(history[-BASE_BLOCKS:])
    hist_med = _median(history)
    noise = 1.4826 * _median([abs(v - hist_med) for v in history])
    noise = max(noise, NOISE_FLOOR_UV)

    dev = [v - baseline for v in window]
    z = [d / noise for d in dev]
    mean_z = sum(z) / len(z)
    std_z = math.sqrt(sum((v - mean_z) ** 2 for v in z) / len(z))
    feats = [
        mean_z,
        min(z),
        max(z),
        std_z,
        max(z) - min(z),
        sum(1 for v in z if abs(v) > 3.0) / len(z),
        z[-1] - z[0],
        (sum(dev) / len(dev)) / 100.0,
        noise / 100.0,
    ]
    info = {"baseline_uv": baseline, "noise_uv": noise,
            "deviation_uv": dev[-1], "peak_deviation_uv": max(dev, key=abs)}
    return feats, info


def blocks_from_samples(samples):
    """10-Hz samples -> 0.5-s block averages (drops an unfinished last block)."""
    out = []
    for i in range(0, len(samples) - BLOCK + 1, BLOCK):
        chunk = [v for v in samples[i:i + BLOCK] if v is not None]
        if chunk:
            out.append(sum(chunk) / len(chunk))
        elif out:
            out.append(out[-1])
    return out


class LogisticModel:
    """Tiny logistic-regression model stored as plain JSON (no pickle, works on any
    Python / scikit-learn version)."""

    def __init__(self, data):
        self.data = data
        self.mean = data["scaler_mean"]
        self.scale = data["scaler_scale"]
        self.coef = data["coef"]
        self.intercept = data["intercept"]
        self.threshold = data.get("threshold", 0.5)

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def probability(self, feats):
        s = self.intercept
        for x, m, sc, w in zip(feats, self.mean, self.scale, self.coef):
            s += w * (x - m) / (sc if sc else 1.0)
        s = max(min(s, 50.0), -50.0)
        return 1.0 / (1.0 + math.exp(-s))


class LiveDetector:
    """Feed it dc_uv samples as they arrive; ask it for the current state."""

    ON_STEPS = 2          # 2 blocks (1 s) above threshold -> event starts
    OFF_STEPS = 4         # 2 s below the lower threshold -> event ends
    LOW_FACTOR = 0.6      # lower threshold = threshold * this (hysteresis)
    RECOVER_S = 60        # "Recovering" lasts this long after an event
    MAX_LOG = 30

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model = None
        self.model_error = None
        try:
            self.model = LogisticModel.load(model_path)
        except Exception as e:      # dashboard still works, it just says the model is missing
            self.model_error = f"{type(e).__name__}: {e}"
        self.reset()

    def reset(self):
        """Forget everything, including the event log."""
        self.events = deque(maxlen=self.MAX_LOG)           # finished events, newest last
        self.event_count = 0
        self.reset_signal()

    def reset_signal(self):
        """Forget the signal history (e.g. after the ESP32 restarted or was offline)
        but keep the event log."""
        self.pending = []                                   # samples not yet in a block
        self.blocks = deque(maxlen=HISTORY_BLOCKS + 40)
        self.state = "warming_up" if getattr(self, "model", None) is not None else "no_model"
        self.prob = 0.0
        self.info = {}
        self.above = 0
        self.below = 0
        if getattr(self, "event", None):                  # close an unfinished event
            self.event["ended"] = self.last_update
            self.event["duration_s"] = round((self.last_update or 0) - self.event["started"], 1)
            self.events.append(self.event)
        self.event = None                                  # the event happening now
        self.recover_until = None
        self.last_update = None
        self.last_seq = None

    # ---- input ----------------------------------------------------------
    MAX_GAP_S = 10        # longer silence / missing data -> start the history again

    def push(self, dc_uv_list, seq_list=None, now=None):
        """dc_uv_list: new samples (microvolts), oldest first. None values are skipped.
        seq_list (optional): the ESP32 sample numbers, used to skip re-sent samples
        and to notice restarts and long gaps."""
        now = time.time() if now is None else now
        if self.last_update is not None and now - self.last_update > self.MAX_GAP_S:
            self.reset_signal()                     # ESP32 was silent too long
        for i, v in enumerate(dc_uv_list):
            if seq_list is not None and i < len(seq_list):
                seq = seq_list[i]
                if self.last_seq is not None:
                    if seq < self.last_seq - 30 * SAMPLE_RATE:
                        self.reset_signal()         # sample numbers started again -> ESP32 restarted
                    elif seq <= self.last_seq:
                        continue                    # already seen (re-sent or late batch)
                    elif seq - self.last_seq > self.MAX_GAP_S * SAMPLE_RATE:
                        self.reset_signal()         # too many samples missing
                self.last_seq = seq
            if v is None:
                continue
            self.pending.append(float(v))
            if len(self.pending) >= BLOCK:
                self.blocks.append(sum(self.pending) / len(self.pending))
                self.pending = []
                self._step(now)
        self.last_update = now

    # ---- one 0.5-s step ----------------------------------------------------
    def _step(self, now):
        if self.model is None or len(self.blocks) < HISTORY_BLOCKS:
            self.state = "warming_up" if self.model is not None else "no_model"
            return
        feats, info = features_from_blocks(self.blocks)
        p = self.model.probability(feats)
        self.prob, self.info = p, info
        thr = self.model.threshold
        low = thr * self.LOW_FACTOR

        if self.state in ("warming_up", "no_model"):
            self.state = "calm"

        if self.state == "event":
            if abs(info["peak_deviation_uv"]) > abs(self.event["peak_uv"]):
                self.event["peak_uv"] = info["peak_deviation_uv"]
            self.event["max_prob"] = max(self.event["max_prob"], p)
            self.below = self.below + 1 if p < low else 0
            if self.below >= self.OFF_STEPS:
                self.event["ended"] = now
                self.event["duration_s"] = round(now - self.event["started"], 1)
                self.events.append(self.event)
                self.event = None
                self.state = "recovering"
                self.recover_until = now + self.RECOVER_S
                self.above = 0
            return

        # calm or recovering: watch for a new event
        self.above = self.above + 1 if p >= thr else 0
        if self.above >= self.ON_STEPS:
            self.event_count += 1
            self.event = {"id": self.event_count, "started": now, "ended": None,
                          "duration_s": None, "peak_uv": info["peak_deviation_uv"],
                          "max_prob": p}
            self.state = "event"
            self.below = 0
            return
        if self.state == "recovering" and now >= self.recover_until:
            self.state = "calm"

    # ---- output ------------------------------------------------------------
    def status(self, now=None):
        now = time.time() if now is None else now
        labels = {
            "warming_up": "Warming up",
            "no_model": "Detector model missing",
            "calm": "Calm",
            "event": "Electrical disturbance detected",
            "recovering": "Recovering",
        }
        needed = HISTORY_BLOCKS * BLOCK_S
        have = len(self.blocks) * BLOCK_S
        out = {
            "state": self.state,
            "label": labels[self.state],
            "probability": round(self.prob, 3),
            "threshold": self.model.threshold if self.model else None,
            "baseline_uv": round(self.info.get("baseline_uv", 0.0), 1) if self.info else None,
            "noise_uv": round(self.info.get("noise_uv", 0.0), 1) if self.info else None,
            "deviation_uv": round(self.info.get("deviation_uv", 0.0), 1) if self.info else None,
            "warmup_left_s": max(0, round(needed - have)),
            "events_total": self.event_count,
            "current_event": self._event_out(self.event, now) if self.event else None,
            "recent_events": [self._event_out(e, now) for e in reversed(self.events)][:10],
            "recover_left_s": (max(0, round(self.recover_until - now))
                               if self.state == "recovering" and self.recover_until else None),
            "model": ({"trained_on": self.model.data.get("trained_on"),
                       "evaluation": self.model.data.get("evaluation")}
                      if self.model else {"error": self.model_error}),
        }
        return out

    @staticmethod
    def _event_out(e, now):
        return {
            "id": e["id"],
            "started": e["started"],
            "ended": e["ended"],
            "duration_s": e["duration_s"] if e["ended"] else round(now - e["started"], 1),
            "peak_mv": round(e["peak_uv"] / 1000.0, 3),
            "confidence": round(e["max_prob"], 2),
        }
