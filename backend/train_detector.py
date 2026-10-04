"""
PhytoSense - train the live event detector from your recorded sessions
=====================================================================

Run (from the backend folder):
    python train_detector.py
    python train_detector.py --csv-dir "D:\\PhytoSense_data\\csv"

What it does
------------
1. Reads every session CSV in the folder (the files saved automatically when you press Stop).
2. Turns each session into "features" every 0.5 s, using detector.py (the same code the
   live dashboard uses).
3. Labels: in touch / pinch sessions the stimulus was given at 2:00, so the seconds
   2:01 - 2:20 are "event"; everything far from 2:00 (and all baseline sessions) is "calm".
4. HONEST TEST (leave-one-session-out): for each session, a model is trained on all the
   OTHER sessions and then the held-out session is replayed through the live detector,
   exactly like on the dashboard. We count:
       hits         = touch/pinch sessions where an event was detected near 2:00
       false alarms = events detected when nothing was done
5. Trains the final model on all sessions and saves it to models/event_detector.json
   plus a readable report in models/training_report.txt.
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import detector as D

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

# Sessions that must NOT be used, with the reason (matched against the note in the file name)
EXCLUDE_NOTES = {
    "sample-testing": "setup test before the experiment",
    "touch-1": "touched the electrode (pad) leaf itself -> electrode artifact, not a plant signal",
    "pinch-1": "pinched the electrode (pad) leaf itself -> electrode artifact, not a plant signal",
}
# Notes typed by mistake -> what they really were
NOTE_FIXES = {"mechanical-3": "touch-3"}

STIM_S = 120.0                 # stimulus given at 2:00
POS = (121.0, 140.0)           # window END inside this range = "event"
IGNORE = (112.0, 165.0)        # around the stimulus but not clearly event -> not used for training
HIT_RANGE = (119.0, 150.0)     # a detected event starting here counts as a hit
QUIET_RANGE = (112.0, 170.0)   # in stimulus sessions, events outside this are false alarms
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


# ---- loading ------------------------------------------------------------------
def parse_name(path):
    name = os.path.basename(path)
    m = re.match(r"phytosense_session(\d+)_([a-z]+)_?(.*)\.csv$", name)
    if not m:
        return None
    sid, label, note = int(m.group(1)), m.group(2), m.group(3)
    note = NOTE_FIXES.get(note, note)
    kind = "other"
    for k in ("baseline", "touch", "pinch", "water"):
        if note.startswith(k):
            kind = k
    return {"id": sid, "label": label, "note": note, "kind": kind, "file": name}


def load_samples(path):
    """Returns dc_uv samples ordered by sample number, duplicates removed, gaps filled with None."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("dc_uv") in (None, ""):
                continue
            seq = int(r["seq"]) if r.get("seq") not in (None, "") else len(rows)
            rows[seq] = float(r["dc_uv"])
    if not rows:
        return []
    seqs = sorted(rows)
    first, last = seqs[0], seqs[-1]
    return [rows.get(s) for s in range(first, last + 1)]


def session_blocks(samples):
    """Block averages + the time (s from start) at the END of each block."""
    blocks = D.blocks_from_samples(samples)
    times = [(i + 1) * D.BLOCK_S for i in range(len(blocks))]
    return blocks, times


def session_features(blocks, times):
    X, T = [], []
    for i in range(D.HISTORY_BLOCKS, len(blocks) + 1):
        f, _ = D.features_from_blocks(blocks[:i])
        X.append(f)
        T.append(times[i - 1])
    return np.array(X), np.array(T)


def labels_for(kind, T):
    y = np.zeros(len(T), dtype=int)
    use = np.ones(len(T), dtype=bool)
    if kind in ("touch", "pinch"):
        pos = (T >= POS[0]) & (T <= POS[1])
        y[pos] = 1
        use[(T >= IGNORE[0]) & (T <= IGNORE[1]) & ~pos] = False
    return y, use


# ---- models -------------------------------------------------------------------
def fit_logistic(X, y):
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000)
    lr.fit(scaler.transform(X), y)
    return {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def replay(model_dict, threshold, blocks, times):
    """Runs the real live detector over a recorded session. Returns event start times."""
    det = D.LiveDetector.__new__(D.LiveDetector)
    det.model = D.LogisticModel(dict(model_dict, threshold=threshold))
    det.model_error = None
    det.reset()
    starts = []
    for b, t in zip(blocks, times):
        before = det.event_count
        det.blocks.append(b)
        det._step(t)
        if det.event_count > before:
            starts.append(t)
    return starts


def score_session(kind, starts, duration_s):
    hit = None
    fa = 0
    if kind in ("touch", "pinch"):
        hit = any(HIT_RANGE[0] <= s <= HIT_RANGE[1] for s in starts)
        fa = sum(1 for s in starts if not (QUIET_RANGE[0] <= s <= QUIET_RANGE[1]))
        quiet_s = duration_s - (QUIET_RANGE[1] - QUIET_RANGE[0])
    else:
        fa = len(starts)
        quiet_s = duration_s - D.HISTORY_BLOCKS * D.BLOCK_S
    return hit, fa, max(quiet_s, 0.0)


# ---- main ---------------------------------------------------------------------
def main():
    default_dir = "D:\\PhytoSense_data\\csv" if os.name == "nt" else os.path.join(HERE, "data", "csv")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=default_dir)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.csv_dir, "phytosense_session*.csv")))
    if not files:
        sys.exit(f"No session CSV files found in {args.csv_dir}")

    sessions, skipped = [], []
    for path in files:
        meta = parse_name(path)
        if meta is None:
            continue
        if meta["note"] in EXCLUDE_NOTES:
            skipped.append((meta, EXCLUDE_NOTES[meta["note"]]))
            continue
        if meta["kind"] == "other":
            skipped.append((meta, "unknown note (expected baseline / touch / pinch / water)"))
            continue
        samples = load_samples(path)
        blocks, times = session_blocks(samples)
        if len(blocks) < D.HISTORY_BLOCKS + 10:
            skipped.append((meta, "too short"))
            continue
        X, T = session_features(blocks, times)
        y, use = labels_for(meta["kind"], T)
        meta.update(blocks=blocks, times=times, X=X, T=T, y=y, use=use,
                    duration_s=times[-1])
        sessions.append(meta)
    sessions.sort(key=lambda s: s["id"])

    train_kinds = ("baseline", "touch", "pinch")
    train_sessions = [s for s in sessions if s["kind"] in train_kinds]
    extra_sessions = [s for s in sessions if s["kind"] not in train_kinds]   # e.g. watering
    if not any(s["kind"] in ("touch", "pinch") for s in train_sessions):
        sys.exit("Need at least one touch or pinch session to train.")

    lines = []
    say = lambda msg="": (print(msg), lines.append(msg))
    say("PhytoSense event detector - training report")
    say(f"Created: {dt.datetime.now():%Y-%m-%d %H:%M}")
    say(f"CSV folder: {args.csv_dir}")
    say()
    say(f"Sessions used for training: {len(train_sessions)}   "
        f"(baseline {sum(s['kind']=='baseline' for s in train_sessions)}, "
        f"touch {sum(s['kind']=='touch' for s in train_sessions)}, "
        f"pinch {sum(s['kind']=='pinch' for s in train_sessions)})")
    for meta, why in skipped:
        say(f"  skipped session {meta['id']} ({meta['note']}): {why}")
    say()

    # ---- leave-one-session-out --------------------------------------------------
    per_thr = {t: {"hits": 0, "events": 0, "fa": 0, "quiet_s": 0.0, "rows": []} for t in THRESHOLDS}
    rf_correct = rf_total = 0
    for held in train_sessions:
        others = [s for s in train_sessions if s is not held]
        Xtr = np.vstack([s["X"][s["use"]] for s in others])
        ytr = np.concatenate([s["y"][s["use"]] for s in others])
        model = fit_logistic(Xtr, ytr)
        for t in THRESHOLDS:
            starts = replay(model, t, held["blocks"], held["times"])
            hit, fa, quiet = score_session(held["kind"], starts, held["duration_s"])
            r = per_thr[t]
            if hit is not None:
                r["events"] += 1
                r["hits"] += int(hit)
            r["fa"] += fa
            r["quiet_s"] += quiet
            r["rows"].append((held, starts, hit, fa))
        # random forest, window-level accuracy only (for comparison)
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                    random_state=0).fit(Xtr, ytr)
        m = held["use"]
        rf_correct += int((rf.predict(held["X"][m]) == held["y"][m]).sum())
        rf_total += int(m.sum())

    say("Honest test: leave-one-session-out, replayed through the live detector")
    say("threshold | events detected | false alarms | false alarms per hour of quiet time")
    best_t, best_score = None, None
    for t in THRESHOLDS:
        r = per_thr[t]
        fa_h = r["fa"] / (r["quiet_s"] / 3600.0) if r["quiet_s"] else 0.0
        score = r["hits"] - 2 * r["fa"]
        say(f"   {t:.1f}    |     {r['hits']:2d} / {r['events']:2d}     |      {r['fa']:2d}      |   {fa_h:5.1f}")
        if best_score is None or score > best_score:
            best_t, best_score = t, score
    say(f"Chosen threshold: {best_t}")
    say()
    r = per_thr[best_t]
    say(f"Per session at threshold {best_t}:")
    for held, starts, hit, fa in r["rows"]:
        when = ", ".join(f"{int(s // 60)}:{int(s % 60):02d}" for s in starts) or "-"
        verdict = ("HIT" if hit else "MISSED") if hit is not None else ("ok" if fa == 0 else "FALSE ALARM")
        say(f"  session {held['id']:2d} {held['note']:<12} events at: {when:<20} {verdict}"
            + (f"  ({fa} false alarm)" if hit is not None and fa else ""))
    say()

    # ---- final model on everything ----------------------------------------------
    Xall = np.vstack([s["X"][s["use"]] for s in train_sessions])
    yall = np.concatenate([s["y"][s["use"]] for s in train_sessions])
    final = fit_logistic(Xall, yall)

    for s in extra_sessions:
        starts = replay(final, best_t, s["blocks"], s["times"])
        when = ", ".join(f"{int(x // 60)}:{int(x % 60):02d}" for x in starts) or "none"
        say(f"Not used for training - session {s['id']} ({s['note']}): events detected at {when}")
    if extra_sessions:
        say()

    say(f"Random forest (comparison only, window accuracy): {rf_correct / max(rf_total, 1):.1%}")
    say("Kept the logistic regression: similar results, simple, explainable, saved as plain JSON.")
    say()
    say("Model weights (positive = pushes towards 'event'):")
    for name, w in sorted(zip(D.FEATURE_NAMES, final["coef"]), key=lambda p: -abs(p[1])):
        say(f"  {name:<13} {w:+.2f}")

    hits, events = per_thr[best_t]["hits"], per_thr[best_t]["events"]
    fa_h = per_thr[best_t]["fa"] / (per_thr[best_t]["quiet_s"] / 3600.0) if per_thr[best_t]["quiet_s"] else 0.0
    model_out = dict(final)
    model_out.update({
        "type": "logistic_regression",
        "feature_names": D.FEATURE_NAMES,
        "threshold": best_t,
        "trained_on": {
            "date": f"{dt.date.today()}",
            "sessions": len(train_sessions),
            "baseline": sum(s["kind"] == "baseline" for s in train_sessions),
            "touch": sum(s["kind"] == "touch" for s in train_sessions),
            "pinch": sum(s["kind"] == "pinch" for s in train_sessions),
        },
        "evaluation": {
            "method": "leave-one-session-out, replayed through live detector",
            "events_detected": hits, "events_total": events,
            "false_alarms": per_thr[best_t]["fa"],
            "false_alarms_per_hour": round(fa_h, 1),
        },
    })
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "event_detector.json"), "w", encoding="utf-8") as f:
        json.dump(model_out, f, indent=2)
    with open(os.path.join(MODEL_DIR, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print()
    print(f"Saved: {os.path.join(MODEL_DIR, 'event_detector.json')}")
    print(f"Saved: {os.path.join(MODEL_DIR, 'training_report.txt')}")


if __name__ == "__main__":
    main()
