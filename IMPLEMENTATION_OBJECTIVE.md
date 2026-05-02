# Project Implementation Objective

Source: `akin 1-3.docx` (Chapter One, Section 1.2 "Aim and Objectives")

## Aim

Develop and evaluate an AI-powered virtual health assistant for real-time patient monitoring and support.

## Core Objectives (from the document)

1. Design an intelligent system for patient interaction.
2. Develop a real-time monitoring framework.
3. Implement machine learning models for health prediction.
4. Evaluate system performance using standard metrics.
5. Compare the system with existing healthcare solutions.

## Implementation-Focused Interpretation (for this codebase)

- Build a usable virtual assistant interface for patient Q&A and guidance.
- Continuously track core vitals (heart rate, temperature, oxygen, blood pressure).
- Detect abnormal patterns and generate timely alerts/recommendations.
- Include AI-driven response logic (LLM or rule-based fallback for MVP).
- Measure practical performance (stability, responsiveness, alert correctness, UX flow).
- Keep the MVP modular so it can later connect to wearables, EHR, and richer models.

## Current Codebase Mapping

- `app.py` is the main Streamlit dashboard and chatbot demo.
- `archive/app_experiment.py` is an experimental/test Streamlit variant retained for reference.
- `backend/app/main.py` exposes ingestion, simulation, upload, chat, prediction, summary, and evaluation APIs.
- `backend/app/alert_engine.py` contains the explainable rule-based alert baseline.
- `backend/app/prediction_engine.py` adds an MVP prediction layer with risk score, risk level, trends, and next-value estimates.
- `backend/app/ml_model.py` loads a trained Gaussian Naive Bayes baseline for normal/warning/critical risk classification.
- `backend/scripts/train_health_model.py` generates a labeled synthetic vitals dataset and trains the baseline model.
- `backend/app/evaluation.py` reports precision, recall, and F1 for embedded demo alert cases.
- `backend/ml/model_metrics.json` reports trained model accuracy, macro-F1, and per-class metrics.
- `PROJECT_ALIGNMENT.md` explains how the implementation maps to the five objectives and what remains for a full study.

## Scope Reminder

- Non-critical healthcare support only.
- Not a replacement for clinical diagnosis or emergency care.

