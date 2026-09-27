"""
dsp_filters.py
--------------
Cleans the plant signal before it is shown on the dashboard and
before features are extracted for the classifier.

Steps (in this order):
  1. Mains-hum notch filter  -> only for the 100 Hz simulator.
     The real ESP32 already cancels 50 Hz hum by averaging 100 readings
     over exactly 100 ms, so its 10 Hz data does not need this.
  2. Baseline removal        -> subtracts the resting level of the
     AD8232 output (~760 mV on your board), so the signal is shown as
     "change from baseline" around 0 mV.
  3. Moving-average smoothing -> removes small ADC noise. The ends are
     padded with copies of the edge values, so the first and last
     points are no longer dragged down towards zero.
"""
import numpy as np
from scipy import signal


class BioSignalDSP:
    def __init__(self, mains_freq=50.0, quality_factor=30.0, smooth_seconds=0.3):
        self.mains_freq = mains_freq
        self.quality_factor = quality_factor
        self.smooth_seconds = smooth_seconds

    def apply_notch(self, data, sample_rate):
        # A notch filter can only remove frequencies below half the sample
        # rate (the Nyquist limit). At 10 samples/s nothing at 50 Hz can exist
        # in the data, so we skip it.
        if self.mains_freq >= sample_rate / 2 or len(data) < 15:
            return data
        b, a = signal.iirnotch(w0=self.mains_freq, Q=self.quality_factor, fs=sample_rate)
        return signal.filtfilt(b, a, data)

    def remove_baseline(self, data):
        # The median is not thrown off by a single spike, unlike the mean.
        return data - np.median(data)

    def moving_average(self, data, sample_rate):
        window = max(1, int(round(self.smooth_seconds * sample_rate)))
        if window < 2 or len(data) < window:
            return data
        # Pad both ends with the edge values, then keep only the "valid"
        # part, so the output has the same length with no drop at the ends.
        left = window // 2
        right = window - 1 - left
        padded = np.pad(data, (left, right), mode="edge")
        kernel = np.ones(window) / window
        return np.convolve(padded, kernel, mode="valid")

    def process(self, raw_data, sample_rate):
        data = np.asarray(raw_data, dtype=float)
        if data.size == 0:
            return data
        data = self.apply_notch(data, sample_rate)
        data = self.remove_baseline(data)
        return self.moving_average(data, sample_rate)