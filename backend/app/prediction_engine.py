"""
Lightweight prediction and risk scoring for the MVP.

This is intentionally simple and explainable: it is not a clinical model, but it
gives the project a measurable prediction component that can later be replaced
with a trained model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ml_model import predict_ml
from .models import VitalReading


@dataclass(frozen=True)
class PredictionResult:
    risk_score: int
    risk_level: str
    drivers: list[str]
    trends: dict[str, float]
    next_estimate: dict[str, float | None]
    ml_prediction: dict


def _latest_with_metric(readings: list[VitalReading], metric: str) -> list[float]:
    values: list[float] = []
    for reading in readings:
        value = getattr(reading, metric)
        if value is not None:
            values.append(float(value))
    return values


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round((values[-1] - values[0]) / (len(values) - 1), 3)


def _next(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[-1], 2)
    return round(values[-1] + _trend(values[-6:]), 2)


def predict_recent(readings: list[VitalReading]) -> PredictionResult:
    ordered = sorted(readings, key=lambda r: r.recorded_at)
    metrics = {
        "heart_rate": _latest_with_metric(ordered, "heart_rate"),
        "spo2": _latest_with_metric(ordered, "spo2"),
        "bp_systolic": _latest_with_metric(ordered, "bp_systolic"),
        "bp_diastolic": _latest_with_metric(ordered, "bp_diastolic"),
        "body_temp_c": _latest_with_metric(ordered, "body_temp_c"),
    }

    trends = {name: _trend(values[-6:]) for name, values in metrics.items()}
    next_estimate = {name: _next(values[-6:]) for name, values in metrics.items()}
    drivers: list[str] = []
    score = 0.0

    latest = ordered[-1] if ordered else None
    if latest is None:
        return PredictionResult(
            0,
            "unknown",
            ["No readings available"],
            trends,
            next_estimate,
            predict_ml([]),
        )

    if latest.heart_rate is not None:
        if latest.heart_rate > 120 or latest.heart_rate < 50:
            score += 28
            drivers.append("heart rate outside expected demo range")
        elif latest.heart_rate > 100 or latest.heart_rate < 60:
            score += 14
            drivers.append("heart rate nearing alert range")
        if abs(trends["heart_rate"]) >= 3:
            score += 6
            drivers.append("heart rate trend is changing quickly")

    if latest.spo2 is not None:
        if latest.spo2 < 90:
            score += 34
            drivers.append("oxygen saturation is critically low")
        elif latest.spo2 < 95:
            score += 20
            drivers.append("oxygen saturation is below preferred range")
        if trends["spo2"] <= -0.5:
            score += 8
            drivers.append("oxygen saturation is trending downward")

    if latest.bp_systolic is not None:
        if latest.bp_systolic >= 180:
            score += 24
            drivers.append("systolic blood pressure is very high")
        elif latest.bp_systolic >= 140:
            score += 12
            drivers.append("systolic blood pressure is elevated")

    if latest.bp_diastolic is not None:
        if latest.bp_diastolic >= 110:
            score += 16
            drivers.append("diastolic blood pressure is very high")
        elif latest.bp_diastolic >= 90:
            score += 8
            drivers.append("diastolic blood pressure is elevated")

    if latest.body_temp_c is not None:
        if latest.body_temp_c >= 39.5 or latest.body_temp_c < 35:
            score += 22
            drivers.append("temperature is outside expected demo range")
        elif latest.body_temp_c >= 38:
            score += 12
            drivers.append("temperature suggests possible fever")

    risk_score = max(0, min(100, round(score)))
    if risk_score >= 70:
        level = "critical"
    elif risk_score >= 40:
        level = "high"
    elif risk_score >= 15:
        level = "moderate"
    else:
        level = "low"

    return PredictionResult(
        risk_score=risk_score,
        risk_level=level,
        drivers=drivers or ["No major risk drivers detected"],
        trends=trends,
        next_estimate=next_estimate,
        ml_prediction=predict_ml(ordered),
    )
