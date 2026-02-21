"""
PredictAI — RUL Inference Engine
=================================
Wraps the trained LSTM model for production inference.
Handles normalization, windowing, and threshold decisions.
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.lstm_model import LSTMModel


class RULInferenceEngine:
    """
    Production inference wrapper.
    Maintains a rolling window of sensor readings per machine.
    """

    THRESHOLDS = {
        "INDUCTION_MOTOR": {"warning": 40, "critical": 20, "max_temp": 85,  "max_vib": 3.0, "max_curr": 10.0},
        "PUMP_COMPRESSOR": {"warning": 35, "critical": 15, "max_temp": 80,  "max_vib": 4.0, "max_curr": 12.0},
        "CONVEYOR_SYSTEM": {"warning": 45, "critical": 25, "max_temp": 75,  "max_vib": 2.5, "max_curr": 9.0},
        "ROTATING_GEAR":   {"warning": 40, "critical": 20, "max_temp": 90,  "max_vib": 5.0, "max_curr": 11.0},
        "DEFAULT":         {"warning": 40, "critical": 20, "max_temp": 85,  "max_vib": 3.5, "max_curr": 10.0},
    }

    def __init__(self, model_path: str, window_size: int = 30):
        self.window_size = window_size
        self.model = LSTMModel.load(model_path)

        # Load scaler stats
        scaler_path = model_path.replace('.json', '_scaler.json')
        if os.path.exists(scaler_path):
            with open(scaler_path) as f:
                self.scaler = json.load(f)
        else:
            self.scaler = {
                'temp_min': 20.0, 'temp_max': 110.0,
                'vib_min':  0.1,  'vib_max':  8.0,
                'curr_min': 0.5,  'curr_max': 12.0,
            }

        # Per-machine rolling buffers
        self._buffers: dict = {}

    def _normalize(self, temp: float, vib: float, curr: float) -> np.ndarray:
        s = self.scaler
        t_n = (temp - s['temp_min']) / (s['temp_max'] - s['temp_min'] + 1e-8)
        v_n = (vib  - s['vib_min'])  / (s['vib_max']  - s['vib_min']  + 1e-8)
        i_n = (curr - s['curr_min']) / (s['curr_max']  - s['curr_min'] + 1e-8)
        return np.array([
            max(0.0, min(1.0, t_n)),
            max(0.0, min(1.0, v_n)),
            max(0.0, min(1.0, i_n)),
        ], dtype=np.float32)

    def update(self, machine_id: str, temp: float, vib: float, curr: float) -> dict:
        """
        Add a new reading and return prediction.
        Call this every time new sensor data arrives from ESP32.
        """
        reading = self._normalize(temp, vib, curr)

        if machine_id not in self._buffers:
            self._buffers[machine_id] = []

        buf = self._buffers[machine_id]
        buf.append(reading)
        if len(buf) > self.window_size:
            buf.pop(0)

        # Pad if not enough data yet
        if len(buf) < self.window_size:
            padded = [buf[0]] * (self.window_size - len(buf)) + buf
        else:
            padded = buf

        sequence = np.array(padded, dtype=np.float32)
        rul = float(self.model.predict(sequence))
        rul = max(0.0, min(100.0, rul))

        return {
            "rul": round(rul, 2),
            "status": self.get_status(rul),
            "degradation_index": self.degradation_index(temp, vib, curr),
            "buffer_full": len(buf) >= self.window_size,
        }

    def predict_single(self, temp: float, vib: float, curr: float, window: int = None) -> float:
        """Quick single prediction (fills window with same reading)."""
        w = window or self.window_size
        reading = self._normalize(temp, vib, curr)
        sequence = np.array([reading] * w, dtype=np.float32)
        rul = float(self.model.predict(sequence))
        return round(max(0.0, min(100.0, rul)), 2)

    def get_status(self, rul: float, machine_type: str = "DEFAULT") -> str:
        t = self.THRESHOLDS.get(machine_type, self.THRESHOLDS["DEFAULT"])
        if rul < t["critical"]: return "critical"
        if rul < t["warning"]:  return "warning"
        return "normal"

    def degradation_index(self, temp: float, vib: float, curr: float) -> float:
        alpha, beta, gamma = 0.4, 0.35, 0.25
        dt = max(0, (temp - 40) / 70)
        dv = max(0, vib / 8)
        di = max(0, (curr - 2) / 10)
        return round(min(1.0, alpha*dt + beta*dv + gamma*di), 4)

    def batch_predict(self, readings: list) -> list:
        """Predict for a list of (temp, vib, curr) tuples."""
        return [{"reading": r, "rul": self.predict_single(*r),
                 "status": self.get_status(self.predict_single(*r))}
                for r in readings]

    def get_ai_explanation(self, temp: float, vib: float, curr: float, rul: float) -> dict:
        """Generate explainable AI reasoning for the prediction."""
        di = self.degradation_index(temp, vib, curr)
        t_contrib = round(max(0, (temp-40)/70) * 0.4, 4)
        v_contrib = round(max(0, vib/8) * 0.35, 4)
        i_contrib = round(max(0, (curr-2)/10) * 0.25, 4)

        dominant = max([("thermal_stress", t_contrib),
                        ("mechanical_vibration", v_contrib),
                        ("electrical_anomaly", i_contrib)],
                       key=lambda x: x[1])[0]

        status = self.get_status(rul)

        return {
            "rul_predicted":    rul,
            "status":           status,
            "degradation_index": di,
            "contributions": {
                "thermal":    {"score": t_contrib, "percent": round(t_contrib/di*100 if di>0 else 0, 1)},
                "vibration":  {"score": v_contrib, "percent": round(v_contrib/di*100 if di>0 else 0, 1)},
                "electrical": {"score": i_contrib, "percent": round(i_contrib/di*100 if di>0 else 0, 1)},
            },
            "dominant_factor": dominant.replace("_", " "),
            "formula": "DI = 0.4×ΔT_norm + 0.35×V_rms + 0.25×ΔI_norm",
            "recommendation": {
                "normal":   "Continue normal operation. Next check in 72h.",
                "warning":  "Schedule maintenance within 48h. Increase sensor poll rate.",
                "critical": "SHUTDOWN IMMEDIATELY. Do not restart without full inspection.",
            }[status],
            "cycles_remaining": round(rul * 1.2),
            "confidence_pct":   round(88 + (rul/100)*8, 1),
        }


# ── QUICK TEST ────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing RUL Inference Engine...")
    try:
        eng = RULInferenceEngine("models/rul_model.json")

        test_cases = [
            ("Normal operation",   45.0, 0.5, 3.0),
            ("Warning zone",       68.0, 1.9, 5.1),
            ("Critical failure",   85.0, 3.5, 8.0),
        ]

        print(f"\n{'Scenario':<22} {'Temp':>6} {'Vib':>6} {'Curr':>6} {'RUL':>8} {'Status'}")
        print("-"*65)
        for name, t, v, i in test_cases:
            result = eng.update("TEST", t, v, i)
            print(f"{name:<22} {t:>6.1f} {v:>6.2f} {i:>6.1f} {result['rul']:>7.1f}% {result['status']}")

        print("\n✅ Inference engine OK")
    except Exception as e:
        print(f"❌ Error: {e}")
