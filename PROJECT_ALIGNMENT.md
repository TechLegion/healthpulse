# Project Alignment Notes

Source objective: develop and evaluate an AI-powered virtual health assistant
for real-time patient monitoring and support.

## Objective Mapping

1. Design an intelligent system for patient interaction.
   - Implemented by the unified HTML/JS assistant in the `frontend/` directory.
   - Implemented by the backend `POST /api/chat` endpoint.
   - OpenAI is optional; safe fallback responses are always available.

2. Develop a real-time monitoring framework.
   - The frontend dashboard simulates live vitals, updates the UI, and syncs
     readings to the FastAPI backend.
   - Backend endpoints ingest, store, list, upload, and simulate vital readings.

3. Implement machine learning models for health prediction.
   - `backend/app/prediction_engine.py` provides an explainable MVP prediction
     layer: risk score, risk level, trend slopes, and next-value estimates.
   - `backend/scripts/train_health_model.py` generates a labeled synthetic
     vitals dataset and trains a PyTorch Transformer sequence classifier.
   - `backend/app/ml_model.py` loads the saved trained model and returns a
     predicted class, confidence score, and class probabilities.
   - This is a trained baseline model, not a clinical-grade model. It is
     designed so a larger transformer or deep model can replace it later without
     changing the API contract.

4. Evaluate system performance using standard metrics.
   - `backend/app/evaluation.py` evaluates alert rules against embedded demo
     cases and reports precision, recall, and F1 through `GET /api/evaluation`.
   - `backend/ml/model_metrics.json` reports trained model accuracy, macro-F1,
     per-class precision, recall, and F1.
   - These metrics are suitable for MVP demonstration; a final study should use
     a larger real-world labeled dataset.

5. Compare the system with existing healthcare solutions.
   - The codebase now exposes measurable behavior that can be compared:
     alert accuracy, risk-score behavior, response time, and interaction flow.
   - Suggested comparison targets: wearable dashboards, symptom checkers, and
     telehealth triage assistants.

## Recommended Next Research Extensions

- Replace or augment the deterministic risk score with a trained anomaly model.
- Replace or augment the Transformer model with more complex architectures (e.g., Bi-LSTM, GPT-based vitals analysis) trained on a larger real-world dataset.
- Add a real-world labeled CSV dataset for evaluation beyond the embedded demo
  and synthetic cases.
- Persist chat history and assistant safety outcomes for usability analysis.
- Add role-specific views for patient and practitioner workflows.
- Compare dashboard response time and alert quality with two or three existing
  virtual health assistant products.
