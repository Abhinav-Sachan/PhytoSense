"""
Pretends to be the ESP32 so you can test the ingest pipeline before
your hardware arrives. Run this in a SECOND terminal while main.py
is running in the first one.
"""
import time
import random
import math
import requests

BACKEND = "http://127.0.0.1:5000/api/ingest"
TOKEN = "phytosense-dev-token"   # must match DEVICE_TOKEN in main.py

t = 0.0
uptime = 0.0

print("Fake ESP32 started. Sending fake readings every second. Ctrl+C to stop.")

while True:
    batch = []
    for _ in range(10):  # 10 samples/second, matching the real firmware later
        bio_signal = 15 * math.sin(2 * math.pi * 0.05 * t)
        noise = random.gauss(0, 2)
        mv = 1650 + bio_signal + noise   # AD8232 rests at ~half of 3.3V = 1650 mV
        batch.append(round(mv, 2))
        t += 0.1

    payload = {
        "bio_mv": batch,
        "temp_c": round(24 + random.uniform(-0.5, 0.5), 1),
        "humidity_pct": round(55 + random.uniform(-2, 2), 1),
        "light_pct": round(60 + random.uniform(-5, 5), 1),
        "lead_off": False,
        "rssi": -55,
        "uptime_s": uptime,
    }

    try:
        r = requests.post(BACKEND, json=payload, headers={"X-Device-Token": TOKEN}, timeout=2)
        print("sent", len(batch), "samples ->", r.status_code, r.json())
    except Exception as e:
        print("failed to reach backend:", e)

    uptime += 1
    time.sleep(1)