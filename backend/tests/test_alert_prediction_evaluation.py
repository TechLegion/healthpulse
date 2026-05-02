import datetime as dt

from app.alert_engine import evaluate_vital
from app.evaluation import evaluate_alert_rules
from app.ml_model import load_model_metrics, predict_ml
from app.models import AlertSeverity, VitalReading
from app.prediction_engine import predict_recent


def make_vital(**overrides):
    values = {
        "id": 1,
        "recorded_at": dt.datetime.now(dt.UTC),
        "heart_rate": 78,
        "bp_systolic": 118,
        "bp_diastolic": 76,
        "spo2": 97,
        "body_temp_c": 36.8,
        "source": "test",
    }
    values.update(overrides)
    return VitalReading(**values)


def test_alert_engine_flags_low_oxygen_as_critical_below_88():
    alerts = evaluate_vital(make_vital(spo2=87))

    assert [a.code for a in alerts] == ["SPO2_LOW"]
    assert alerts[0].severity == AlertSeverity.critical.value


def test_prediction_risk_increases_for_multiple_abnormal_readings():
    base_time = dt.datetime.now(dt.UTC)
    readings = [
        make_vital(id=i, recorded_at=base_time + dt.timedelta(minutes=i), heart_rate=92 + i * 5, spo2=96 - i)
        for i in range(1, 7)
    ]
    result = predict_recent(readings)

    assert result.risk_score >= 40
    assert result.risk_level in {"high", "critical"}
    assert any("oxygen" in driver for driver in result.drivers)
    assert result.ml_prediction["available"] is True


def test_evaluation_reports_standard_metrics():
    result = evaluate_alert_rules()

    assert result["case_count"] >= 5
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["f1"] <= 1
    assert all("passed" in case for case in result["cases"])


def test_trained_ml_model_returns_class_probabilities():
    prediction = predict_ml([make_vital(spo2=86, bp_systolic=186, bp_diastolic=118)])
    metrics = load_model_metrics()

    assert prediction["available"] is True
    assert prediction["predicted_class"] in {"normal", "warning", "critical"}
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 0.01
    assert metrics is not None
    assert metrics["test_records"] > 0
