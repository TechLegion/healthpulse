import datetime as dt
from pydantic import BaseModel, Field


class VitalIn(BaseModel):
    heart_rate: float | None = Field(None, ge=20, le=240, description="Heart rate in bpm")
    bp_systolic: float | None = Field(None, ge=50, le=260, description="Systolic blood pressure in mmHg")
    bp_diastolic: float | None = Field(None, ge=30, le=180, description="Diastolic blood pressure in mmHg")
    spo2: float | None = Field(None, ge=50, le=100, description="Oxygen saturation %")
    body_temp_c: float | None = Field(None, ge=25, le=45, description="Body temperature in Celsius")
    recorded_at: dt.datetime | None = None
    source: str = Field("api", min_length=1, max_length=32)


class VitalOut(BaseModel):
    id: int
    recorded_at: dt.datetime
    heart_rate: float | None
    bp_systolic: float | None
    bp_diastolic: float | None
    spo2: float | None
    body_temp_c: float | None
    source: str

    model_config = {"from_attributes": True}


class AlertOut(BaseModel):
    id: int
    created_at: dt.datetime
    vital_id: int
    code: str
    message: str
    severity: str

    model_config = {"from_attributes": True}


class ChatMessageIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    include_recent_vitals: bool = True


class ChatMessageOut(BaseModel):
    reply: str
    used_llm: bool


class ChatHistoryItem(BaseModel):
    id: int
    created_at: dt.datetime
    role: str
    content: str
    used_llm: bool

    model_config = {"from_attributes": True}


class SimulateIn(BaseModel):
    count: int = Field(24, ge=1, le=500, description="Number of generated points")
    interval_minutes: int = Field(5, ge=1, le=120)


class PredictionOut(BaseModel):
    risk_score: int
    risk_level: str
    drivers: list[str]
    trends: dict[str, float]
    next_estimate: dict[str, float | None]
    ml_prediction: dict


class SummaryOut(BaseModel):
    latest_vital: VitalOut | None
    active_alerts: list[AlertOut]
    prediction: PredictionOut
    recommendation: str
