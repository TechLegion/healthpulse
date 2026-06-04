"""
Rule-based alerts for demo / non-clinical use only.
Not a medical device; not for diagnosis or emergency use.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import AlertEvent, AlertSeverity, VitalReading


@dataclass(frozen=True)
class Thresholds:
    hr_min: float = 40.0
    hr_max: float = 140.0
    spo2_min: float = 90.0
    bp_sys_max: float = 200.0
    bp_dia_max: float = 120.0
    temp_c_min: float = 34.0
    temp_c_max: float = 40.0


def evaluate_vital(v: VitalReading, t: Thresholds | None = None) -> list[AlertEvent]:
    t = t or Thresholds()
    alerts: list[AlertEvent] = []

    if v.heart_rate is not None:
        if v.heart_rate < t.hr_min:
            alerts.append(
                AlertEvent(
                    vital_id=v.id,
                    code="HR_LOW",
                    message=f"Heart rate is low ({v.heart_rate:.0f} bpm). Consider rest and hydration; contact a clinician if symptoms persist.",
                    severity=AlertSeverity.warn.value,
                )
            )
        elif v.heart_rate > t.hr_max:
            alerts.append(
                AlertEvent(
                    vital_id=v.id,
                    code="HR_HIGH",
                    message=f"Heart rate is elevated ({v.heart_rate:.0f} bpm). Slow down; seek care if chest pain, severe shortness of breath, or fainting.",
                    severity=AlertSeverity.warn.value,
                )
            )

    if v.spo2 is not None and v.spo2 < t.spo2_min:
        sev = AlertSeverity.critical.value if v.spo2 < 88 else AlertSeverity.warn.value
        alerts.append(
            AlertEvent(
                vital_id=v.id,
                code="SPO2_LOW",
                message=f"Oxygen saturation is low ({v.spo2:.0f}%). This demo may not reflect true readings. Seek medical attention for breathing difficulty or persistent low SpO2.",
                severity=sev,
            )
        )

    if v.bp_systolic is not None and v.bp_systolic > t.bp_sys_max:
        alerts.append(
            AlertEvent(
                vital_id=v.id,
                code="BP_SYS_HIGH",
                message=f"Blood pressure appears very high (systolic {v.bp_systolic:.0f} mmHg). Recheck; seek urgent care for severe symptoms.",
                severity=AlertSeverity.warn.value,
            )
        )

    if v.bp_diastolic is not None and v.bp_diastolic > t.bp_dia_max:
        alerts.append(
            AlertEvent(
                vital_id=v.id,
                code="BP_DIA_HIGH",
                message=f"Diastolic pressure is high ({v.bp_diastolic:.0f} mmHg). Recheck per clinician guidance.",
                severity=AlertSeverity.info.value,
            )
        )

    if v.body_temp_c is not None:
        if v.body_temp_c > t.temp_c_max:
            alerts.append(
                AlertEvent(
                    vital_id=v.id,
                    code="FEVER",
                    message=f"Temperature is elevated ({v.body_temp_c:.1f} C). Rest, fluids, and follow clinician advice for fever management.",
                    severity=AlertSeverity.warn.value,
                )
            )
        elif v.body_temp_c < t.temp_c_min:
            alerts.append(
                AlertEvent(
                    vital_id=v.id,
                    code="HYPOTHERMIA_SUSPECT",
                    message=f"Temperature is very low ({v.body_temp_c:.1f} C). Verify measurement; avoid cold exposure; seek help if unwell.",
                    severity=AlertSeverity.warn.value,
                )
            )

    return alerts
