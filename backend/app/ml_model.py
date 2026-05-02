from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import VitalReading

# Graceful fallback for mock environment if Torch is missing
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module: pass
        def Linear(*args, **kwargs): pass
        def TransformerEncoderLayer(*args, **kwargs): pass
        def TransformerEncoder(*args, **kwargs): pass
        Parameter = lambda x: x
    class torch:
        float32 = "float32"
        long = "long"
        @staticmethod
        def randn(*args): return None
        @staticmethod
        def tensor(data, dtype=None): return data
        @staticmethod
        def load(path, weights_only=True): return {}
        class no_grad:
            def __enter__(self): pass
            def __exit__(self, *args): pass

FEATURES = ["heart_rate", "bp_systolic", "bp_diastolic", "spo2", "body_temp_c"]
CLASSES = ["normal", "warning", "critical"]
SEQ_LEN = 6

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "ml" / "health_risk_model.pth"
METRICS_PATH = BASE_DIR / "ml" / "model_metrics.json"

class HealthTransformer(nn.Module):
    def __init__(self, input_dim=5, num_classes=3, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        if not HAS_TORCH: return
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        if not HAS_TORCH: return None
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

def _default_value(feature: str) -> float:
    defaults = {
        "heart_rate": 78.0,
        "bp_systolic": 118.0,
        "bp_diastolic": 76.0,
        "spo2": 97.0,
        "body_temp_c": 36.8,
    }
    return defaults[feature]

def _feature_vector(vital: VitalReading) -> list[float]:
    return [float(getattr(vital, feature)) if getattr(vital, feature) is not None else _default_value(feature) for feature in FEATURES]

@lru_cache(maxsize=1)
def load_model_metrics() -> dict[str, Any] | None:
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def load_model() -> tuple[HealthTransformer, dict[str, Any]]:
    metrics = load_model_metrics() or {}
    model = HealthTransformer(input_dim=len(FEATURES), num_classes=3)
    if HAS_TORCH and MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
    return model, metrics

def predict_ml(vitals_seq: list[VitalReading]) -> dict[str, Any]:
    model, metrics = load_model()
    if not MODEL_PATH.exists() or len(vitals_seq) == 0:
        return {
            "available": False,
            "model_name": "time_series_transformer",
            "predicted_class": "unknown",
            "confidence": 0.0,
            "probabilities": {},
        }
        
    while len(vitals_seq) < SEQ_LEN:
        vitals_seq.insert(0, vitals_seq[0])
    if len(vitals_seq) > SEQ_LEN:
        vitals_seq = vitals_seq[-SEQ_LEN:]

    if not HAS_TORCH:
        # Provide a mock prediction for local demo without dependencies
        latest = vitals_seq[-1]
        probs = [0.8, 0.1, 0.1]
        predicted_class = "normal"
        if latest.spo2 is not None and latest.spo2 < 92:
            probs = [0.1, 0.2, 0.7]
            predicted_class = "critical"
        elif latest.heart_rate is not None and latest.heart_rate > 105:
            probs = [0.2, 0.6, 0.2]
            predicted_class = "warning"
        
        probabilities = {CLASSES[i]: probs[i] for i in range(len(CLASSES))}
        return {
            "available": True,
            "model_name": "time_series_transformer (mock)",
            "predicted_class": predicted_class,
            "confidence": probabilities[predicted_class],
            "probabilities": probabilities,
        }

    seq_features = [_feature_vector(v) for v in vitals_seq]
    x_tensor = torch.tensor([seq_features], dtype=torch.float32)
    
    normalization = metrics.get("normalization")
    if normalization:
        means = torch.tensor(normalization["means"])
        stds = torch.tensor(normalization["stds"])
        x_tensor = (x_tensor - means) / stds

    with torch.no_grad():
        outputs = model(x_tensor)
        probs_t = torch.softmax(outputs, dim=1).squeeze().tolist()
        
    probabilities = {CLASSES[i]: round(probs_t[i], 4) for i in range(len(CLASSES))}
    predicted_class = max(probabilities, key=probabilities.get)
    
    return {
        "available": True,
        "model_name": "time_series_transformer",
        "predicted_class": predicted_class,
        "confidence": probabilities[predicted_class],
        "probabilities": probabilities,
    }
