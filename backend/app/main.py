import csv
import io
import random
import datetime as dt
from contextlib import asynccontextmanager

from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from . import models
from .alert_engine import evaluate_vital
from .chat_service import generate_reply
from .config import settings
from .database import Base, engine, get_db
from .evaluation import evaluate_alert_rules
from .ml_model import load_model_metrics
from .prediction_engine import PredictionResult, predict_recent
from .schemas import (
    AlertOut,
    ChatHistoryItem,
    ChatMessageIn,
    ChatMessageOut,
    PredictionOut,
    SimulateIn,
    SummaryOut,
    VitalIn,
    VitalOut,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="VHA API",
    description="Virtual Health Assistant - demo API (non-clinical, not for diagnosis).",
    version="0.1.0",
    lifespan=lifespan,
)


origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
# Add "null" for file:// protocol support, and common local development server ports
dev_origins = [
    "null",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
for o in dev_origins:
    if o not in origins:
        origins.append(o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(frontend_path, "index.html"))

class VitalWithAlerts(BaseModel):
    vital: VitalOut
    alerts: list[AlertOut]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/debug")
def debug() -> dict:
    """Verify environment variables are loaded correctly (does NOT expose secret values)."""
    key = settings.openai_api_key or ""
    return {
        "openai_configured": settings.openai_configured,
        "openai_key_prefix": key[:7] + "..." if len(key) > 7 else "(not set)",
        "database_url_driver": settings.database_url.split(":")[0] if settings.database_url else "(not set)",
        "cors_origins": settings.cors_origins,
    }


def _vital_to_out(v: models.VitalReading) -> VitalOut:
    return VitalOut.model_validate(v)


def _prediction_to_out(p: PredictionResult) -> PredictionOut:
    return PredictionOut(
        risk_score=p.risk_score,
        risk_level=p.risk_level,
        drivers=p.drivers,
        trends=p.trends,
        next_estimate=p.next_estimate,
        ml_prediction=p.ml_prediction,
    )


def _recent_vitals(db: Session, limit: int = 24) -> list[models.VitalReading]:
    return list(
        reversed(
            db.execute(
                select(models.VitalReading)
                .order_by(desc(models.VitalReading.recorded_at))
                .limit(limit)
            )
            .scalars()
            .all()
        )
    )


def _context_for_chat(vital: models.VitalReading | None, alerts: list[models.AlertEvent]) -> str | None:
    if vital is None:
        return None
    vital_context = (
        f"heart_rate={vital.heart_rate} bpm, "
        f"bp={vital.bp_systolic}/{vital.bp_diastolic} mmHg, "
        f"spo2={vital.spo2}%, "
        f"body_temp_c={vital.body_temp_c}"
    )
    alert_context = ", ".join(f"{a.code}:{a.severity}" for a in alerts) if alerts else "no active alerts"
    return f"latest vitals: {vital_context}; alerts: {alert_context}"


def _csv_values_are_plausible(
    heart_rate: float | None,
    bp_systolic: float | None,
    bp_diastolic: float | None,
    spo2: float | None,
    body_temp_c: float | None,
) -> bool:
    checks = [
        (heart_rate, 20, 240),
        (bp_systolic, 50, 260),
        (bp_diastolic, 30, 180),
        (spo2, 50, 100),
        (body_temp_c, 25, 45),
    ]
    return all(value is None or low <= value <= high for value, low, high in checks)


@app.post("/api/vitals", response_model=VitalWithAlerts)
def add_vital(body: VitalIn, db: Session = Depends(get_db)) -> VitalWithAlerts:
    r = body.recorded_at or dt.datetime.now(dt.UTC)
    if r.tzinfo is None:
        r = r.replace(tzinfo=dt.UTC)
    v = models.VitalReading(
        heart_rate=body.heart_rate,
        bp_systolic=body.bp_systolic,
        bp_diastolic=body.bp_diastolic,
        spo2=body.spo2,
        body_temp_c=body.body_temp_c,
        recorded_at=r,
        source=body.source,
    )
    db.add(v)
    db.flush()
    for ev in evaluate_vital(v):
        db.add(ev)
    db.commit()
    db.refresh(v)
    alerts = db.execute(
        select(models.AlertEvent).where(models.AlertEvent.vital_id == v.id)
    ).scalars().all()
    return VitalWithAlerts(vital=_vital_to_out(v), alerts=[AlertOut.model_validate(a) for a in alerts])


@app.get("/api/vitals", response_model=list[VitalOut])
def list_vitals(limit: int = 100, db: Session = Depends(get_db)) -> list[VitalOut]:
    if limit < 1 or limit > 1000:
        raise HTTPException(400, "limit must be 1-1000")
    rows = db.execute(
        select(models.VitalReading).order_by(desc(models.VitalReading.recorded_at)).limit(limit)
    ).scalars().all()
    return [_vital_to_out(v) for v in rows]


@app.get("/api/alerts", response_model=list[AlertOut])
def list_alerts(limit: int = 50, db: Session = Depends(get_db)) -> list[AlertOut]:
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit must be 1-500")
    rows = db.execute(
        select(models.AlertEvent).order_by(desc(models.AlertEvent.created_at)).limit(limit)
    ).scalars().all()
    return [AlertOut.model_validate(a) for a in rows]


@app.get("/api/prediction", response_model=PredictionOut)
def prediction(limit: int = 24, db: Session = Depends(get_db)) -> PredictionOut:
    if limit < 2 or limit > 500:
        raise HTTPException(400, "limit must be 2-500")
    return _prediction_to_out(predict_recent(_recent_vitals(db, limit)))


@app.get("/api/summary", response_model=SummaryOut)
def summary(db: Session = Depends(get_db)) -> SummaryOut:
    readings = _recent_vitals(db, 24)
    latest = readings[-1] if readings else None
    active_alerts = []
    if latest is not None:
        active_alerts = db.execute(
            select(models.AlertEvent)
            .where(models.AlertEvent.vital_id == latest.id)
            .order_by(desc(models.AlertEvent.created_at))
        ).scalars().all()

    p = predict_recent(readings)
    if p.risk_level in {"critical", "high"}:
        recommendation = "Review the latest alerts and contact a qualified healthcare professional for concerning symptoms."
    elif p.risk_level == "moderate":
        recommendation = "Continue monitoring and recheck abnormal readings with validated equipment."
    elif p.risk_level == "unknown":
        recommendation = "Add or simulate readings to generate a monitoring summary."
    else:
        recommendation = "Current demo readings do not show major risk drivers. Continue routine monitoring."

    return SummaryOut(
        latest_vital=_vital_to_out(latest) if latest else None,
        active_alerts=[AlertOut.model_validate(a) for a in active_alerts],
        prediction=_prediction_to_out(p),
        recommendation=recommendation,
    )


@app.get("/api/evaluation")
def evaluation() -> dict:
    return {
        "alert_rules": evaluate_alert_rules(),
        "ml_model": load_model_metrics(),
        "comparison": [
            {
                "system": "HealthPulse AI MVP",
                "real_time_monitoring": True,
                "patient_chat": True,
                "ml_prediction": True,
                "explainable_alerts": True,
                "local_demo_ready": True,
            },
            {
                "system": "Typical wearable dashboard",
                "real_time_monitoring": True,
                "patient_chat": False,
                "ml_prediction": "limited",
                "explainable_alerts": "limited",
                "local_demo_ready": False,
            },
            {
                "system": "Typical symptom checker",
                "real_time_monitoring": False,
                "patient_chat": True,
                "ml_prediction": "limited",
                "explainable_alerts": False,
                "local_demo_ready": False,
            },
        ],
    }


def _parse_dt(s: str | None) -> dt.datetime:
    if not s or not str(s).strip():
        return dt.datetime.now(dt.UTC)
    raw = str(s).strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        d = dt.datetime.fromisoformat(raw)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.UTC)
        return d
    except ValueError:
        return dt.datetime.now(dt.UTC)


@app.post("/api/vitals/upload-csv", response_model=dict)
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file")
    data = await file.read()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(400, "CSV has no header row")
    headers = {h.strip().lower(): h for h in reader.fieldnames if h}
    def col(name: str) -> str | None:
        return headers.get(name.lower())

    inserted = 0
    alerts_count = 0
    skipped = 0
    for row in reader:
        def getf(*names: str) -> float | None:
            for n in names:
                key = col(n)
                if key and row.get(key) not in (None, ""):
                    try:
                        return float(str(row[key]).strip())
                    except ValueError:
                        return None
            return None

        hr = getf("heart_rate", "hr", "bpm")
        sys_ = getf("bp_systolic", "systolic", "sys")
        dia = getf("bp_diastolic", "diastolic", "dia")
        spo2 = getf("spo2", "sp_o2", "oxygen")
        temp = getf("body_temp_c", "temperature", "temp_c")
        ts_key = col("recorded_at") or col("timestamp") or col("time")
        ts = _parse_dt(row.get(ts_key) if ts_key else None) if ts_key else dt.datetime.now(dt.UTC)

        if not _csv_values_are_plausible(hr, sys_, dia, spo2, temp):
            skipped += 1
            continue

        v = models.VitalReading(
            heart_rate=hr,
            bp_systolic=sys_,
            bp_diastolic=dia,
            spo2=spo2,
            body_temp_c=temp,
            recorded_at=ts,
            source="csv",
        )
        db.add(v)
        db.flush()
        evs = evaluate_vital(v)
        for ev in evs:
            db.add(ev)
        inserted += 1
        alerts_count += len(evs)
    db.commit()
    return {"rows_imported": inserted, "rows_skipped": skipped, "alerts_generated": alerts_count}


@app.post("/api/vitals/simulate", response_model=dict)
def simulate(body: SimulateIn, db: Session = Depends(get_db)) -> dict:
    now = dt.datetime.now(dt.UTC)
    step = dt.timedelta(minutes=body.interval_minutes)
    hr, spo2 = 78.0, 97.0
    sys_, dia = 118.0, 76.0
    temp = 36.8
    inserted = 0
    alerts_count = 0
    rng = random.Random(42)
    for i in range(body.count):
        t = now - step * (body.count - 1 - i)
        # Small random walk + occasional spike
        hr = max(45, min(145, hr + rng.uniform(-6, 6) + (2 if i % 17 == 0 else 0)))
        spo2 = max(85, min(100, spo2 + rng.uniform(-0.8, 0.8) - (1.5 if i % 23 == 0 else 0)))
        sys_ = max(90, min(190, sys_ + rng.uniform(-4, 4)))
        dia = max(55, min(120, dia + rng.uniform(-3, 3)))
        temp = max(35.0, min(40.0, temp + rng.uniform(-0.2, 0.2)))

        v = models.VitalReading(
            heart_rate=round(hr, 1),
            bp_systolic=round(sys_, 1),
            bp_diastolic=round(dia, 1),
            spo2=round(spo2, 1),
            body_temp_c=round(temp, 2),
            recorded_at=t,
            source="simulated",
        )
        db.add(v)
        db.flush()
        evs = evaluate_vital(v)
        for ev in evs:
            db.add(ev)
        inserted += 1
        alerts_count += len(evs)
    db.commit()
    return {"points": inserted, "alerts_generated": alerts_count}


@app.post("/api/chat", response_model=ChatMessageOut)
def chat(body: ChatMessageIn, db: Session = Depends(get_db)) -> ChatMessageOut:
    latest = None
    alerts: list[models.AlertEvent] = []
    if body.include_recent_vitals:
        latest = db.execute(
            select(models.VitalReading).order_by(desc(models.VitalReading.recorded_at)).limit(1)
        ).scalar_one_or_none()
        if latest is not None:
            alerts = db.execute(
                select(models.AlertEvent)
                .where(models.AlertEvent.vital_id == latest.id)
                .order_by(desc(models.AlertEvent.created_at))
            ).scalars().all()

    db.add(models.ChatMessage(role="user", content=body.message, used_llm=False))

    reply, used = generate_reply(body.message, _context_for_chat(latest, alerts))

    db.add(models.ChatMessage(role="assistant", content=reply, used_llm=used))
    db.commit()

    return ChatMessageOut(reply=reply, used_llm=used)


@app.get("/api/chat/history", response_model=list[ChatHistoryItem])
def chat_history(limit: int = 50, db: Session = Depends(get_db)) -> list[ChatHistoryItem]:
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit must be 1-500")
    rows = db.execute(
        select(models.ChatMessage)
        .order_by(desc(models.ChatMessage.created_at))
        .limit(limit)
    ).scalars().all()
    return [ChatHistoryItem.model_validate(r) for r in reversed(rows)]


@app.delete("/api/chat/history", response_model=dict)
def clear_chat_history(db: Session = Depends(get_db)) -> dict:
    count = db.query(models.ChatMessage).delete()
    db.commit()
    return {"deleted": count}
