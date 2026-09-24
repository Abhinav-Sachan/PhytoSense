import numpy as np
from sklearn.ensemble import RandomForestClassifier

class PlantClassifier:
    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self._train_synthetic_baseline()

    def _train_synthetic_baseline(self):
        # Generates a basic synthetic dataset to initialize the RF model
        X_train = np.random.rand(300, 4) * [5, 10, 3, 0.5] 
        y_train = np.random.choice(["Healthy", "Mild Stress", "Water Deficit"], 300)
        self.model.fit(X_train, y_train)

    def extract_features(self, data):
        data = np.array(data)
        rms = np.sqrt(np.mean(data**2))
        max_amp = np.max(np.abs(data))
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        zcr = len(zero_crossings) / len(data) if len(data) > 0 else 0
        
        # Dominant frequency via FFT
        freqs = np.fft.rfftfreq(len(data), d=1/self.sample_rate)
        fft_mag = np.abs(np.fft.rfft(data))
        dom_freq = freqs[np.argmax(fft_mag)] if len(fft_mag) > 0 else 0
        
        return {"rms": float(rms), "max_amplitude": float(max_amp), 
                "dominant_frequency": float(dom_freq), "zero_crossing_rate": float(zcr)}

    def classify(self, features):
        feature_vector = [[features["rms"], features["max_amplitude"], 
                           features["dominant_frequency"], features["zero_crossing_rate"]]]
        probs = self.model.predict_proba(feature_vector)[0]
        classes = self.model.classes_
        
        prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
        best_idx = np.argmax(probs)
        
        return {
            "label": classes[best_idx],
            "confidence": float(probs[best_idx]),
            "probabilities": prob_dict
        }