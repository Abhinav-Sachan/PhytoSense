import numpy as np
from scipy import signal

class BioSignalDSP:
    def __init__(self, sample_rate=100, mains_freq=50.0, quality_factor=30.0):
        self.sample_rate = sample_rate
        # Design a notch filter. H(z) isolates and removes exactly 50 Hz.
        self.b, self.a = signal.iirnotch(w0=mains_freq, Q=quality_factor, fs=sample_rate)

    def apply_notch(self, data):
        if len(data) < 15:
            return data
        # filtfilt applies the filter forward and backward to ensure zero phase distortion
        return signal.filtfilt(self.b, self.a, data)

    def moving_average(self, data, window_size=5):
        if len(data) < window_size:
            return data
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')
        
    def process(self, raw_data):
        notched = self.apply_notch(raw_data)
        return self.moving_average(notched)