import threading
import time
import numpy as np

class SensorStream:
    """Fake plant signal. Runs until your real hardware takes over, so
    the dashboard never goes blank while you're still wiring things up."""
    def __init__(self, sample_rate=100, buffer_size=200):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.raw_buffer = []
        self.t = 0.0

        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def _read_raw_sample(self):
        mains_hum = 1.5 * np.sin(2 * np.pi * 50 * self.t)
        bio_signal = 0.5 * np.sin(2 * np.pi * 0.5 * self.t)
        noise = np.random.normal(0, 0.2)
        return bio_signal + mains_hum + noise

    def _stream_loop(self):
        sleep_time = 1.0 / self.sample_rate
        while self._running:
            val = self._read_raw_sample()
            with self._lock:
                self.raw_buffer.append(val)
                if len(self.raw_buffer) > self.buffer_size:
                    self.raw_buffer.pop(0)
            self.t += sleep_time
            time.sleep(sleep_time)

    def get_latest_window(self):
        with self._lock:
            return list(self.raw_buffer)


class IngestBuffer:
    """
    Holds real samples posted by the ESP32 over Wi-Fi.
    Flask writes to this from the /api/ingest route; the dashboard
    routes read from it. Thread-safe because both happen at once.
    """
    def __init__(self, max_seconds=20, expected_rate=10):
        self.max_len = max_seconds * expected_rate
        self._lock = threading.Lock()
        self._bio = []          # just the values, oldest first
        self.last_seen = 0.0    # unix time of the last batch received
        self.latest_telemetry = {}

    def push(self, bio_mv_list, telemetry: dict):
        now = time.time()
        with self._lock:
            self._bio.extend(bio_mv_list)
            if len(self._bio) > self.max_len:
                self._bio = self._bio[-self.max_len:]
            self.last_seen = now
            self.latest_telemetry = telemetry

    def get_latest_window(self):
        with self._lock:
            return list(self._bio)

    def is_online(self, timeout=5.0):
        """True if the ESP32 has spoken to us in the last `timeout` seconds."""
        return (time.time() - self.last_seen) < timeout