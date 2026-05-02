"""
Small, reproducible evaluation harness for the MVP alert rules.

The sample set is deliberately embedded so the project can demonstrate standard
metrics without requiring a separate dataset download.
"""
from __future__ import annotations

import datetime as dt

from .alert_engine import evaluate_vital
from .models import VitalReading


EVALUATION_CASES = [
    {
        "name": "normal baseline",
        "vitals": dict(heart_rate=78, bp_systolic=118, bp_diastolic=76, spo2=97, body_temp_c=36.8),
        "expected": set(),
    },
    {
        "name": "high heart rate",
        "vitals": dict(heart_rate=132, bp_systolic=122, bp_diastolic=80, spo2=97, body_temp_c=37.0),
        "expected": {"HR_HIGH"},
    },
    {
        "name": "low oxygen",
        "vitals": dict(heart_rate=88, bp_systolic=124, bp_diastolic=82, spo2=89, body_temp_c=37.1),
        "expected": {"SPO2_LOW"},
    },
    {
        "name": "very high blood pressure",
        "vitals": dict(heart_rate=94, bp_systolic=185, bp_diastolic=112, spo2=96, body_temp_c=36.9),
        "expected": {"BP_SYS_HIGH", "BP_DIA_HIGH"},
    },
    {
        "name": "fever",
        "vitals": dict(heart_rate=96, bp_systolic=126, bp_diastolic=84, spo2=96, body_temp_c=39.7),
        "expected": {"FEVER"},
    },
    {
        "name": "suspected hypothermia",
        "vitals": dict(heart_rate=58, bp_systolic=110, bp_diastolic=70, spo2=98, body_temp_c=34.6),
        "expected": {"HYPOTHERMIA_SUSPECT"},
    },
]


def evaluate_alert_rules() -> dict:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    cases = []

    for index, case in enumerate(EVALUATION_CASES, start=1):
        vital = VitalReading(
            id=index,
            recorded_at=dt.datetime.now(dt.UTC),
            source="evaluation",
            **case["vitals"],
        )
        predicted = {alert.code for alert in evaluate_vital(vital)}
        expected = case["expected"]

        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        cases.append(
            {
                "name": case["name"],
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "passed": fp == 0 and fn == 0,
            }
        )

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "dataset": "embedded_demo_cases",
        "case_count": len(EVALUATION_CASES),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "cases": cases,
    }
