"""
Fast-channel signal numbers (AD8232 waveform).

This file used to contain a "classifier" that was trained on RANDOM numbers and
pretended to tell Healthy / Mild Stress / Water Deficit. That was removed on
3 Oct 2026. The real, trained plant-state detector now lives in detector.py
(it uses the slow ADS1115 channel and was trained on real recordings).

What is left here is honest signal description only: RMS, peak, dominant
frequency and zero-crossing rate of the filtered fast-channel waveform.
"""

import numpy as np


class SignalFeatures:
    def __init__(self, sample_rate=10):
        self.sample_rate = sample_rate

    def extract(self, data):
        data = np.asarray(data, dtype=float)
        if data.size == 0:
            return {"rms": 0.0, "max_amplitude": 0.0, "dominant_frequency": 0.0, "zero_crossing_rate": 0.0}
        rms = float(np.sqrt(np.mean(data ** 2)))
        max_amp = float(np.max(np.abs(data)))
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        zcr = len(zero_crossings) / len(data)
        freqs = np.fft.rfftfreq(len(data), d=1 / self.sample_rate)
        fft_mag = np.abs(np.fft.rfft(data - data.mean()))
        dom_freq = float(freqs[np.argmax(fft_mag)]) if len(fft_mag) > 1 else 0.0
        return {"rms": rms, "max_amplitude": max_amp,
                "dominant_frequency": dom_freq, "zero_crossing_rate": float(zcr)}
