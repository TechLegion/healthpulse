import datetime as dt
from enum import StrEnum

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class AlertSeverity(StrEnum):
    info = "info"
    warn = "warn"
    critical = "critical"


class VitalReading(Base):
    __tablename__ = "vital_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    heart_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    bp_systolic: Mapped[float | None] = mapped_column(Float, nullable=True)
    bp_diastolic: Mapped[float | None] = mapped_column(Float, nullable=True)
    spo2: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="api")


class AlertEvent(Base):
    __tablename__ = "alert_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    vital_id: Mapped[int] = mapped_column(Integer, index=True)
    code: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(16), default=AlertSeverity.warn.value)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    used_llm: Mapped[bool] = mapped_column(Boolean, default=False)
