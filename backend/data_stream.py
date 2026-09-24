import threading
import time
import numpy as np

class SensorStream:
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
        """FUTURE HARDWARE INTEGRATION: Replace this logic with pyserial read."""
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