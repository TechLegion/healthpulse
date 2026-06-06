"""
Conversational layer: NLP Transformer implementation.
Educational / demo use only - not a substitute for professional care.
"""
from __future__ import annotations

import re
import os
import openai
from app.config import settings

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

_nlp_model = None
_nlp_tokenizer = None

DISCLAIMER = (
    "\n\n*This is a demo assistant — not a substitute for professional medical advice.*"
)


def _get_nlp_model():
    """Lazy-load the Flan-T5 model on first chat request."""
    global _nlp_model, _nlp_tokenizer
    if _nlp_model is None and HAS_TRANSFORMERS:
        try:
            _nlp_tokenizer = AutoTokenizer.from_pretrained(
                "google/flan-t5-small", local_files_only=True
            )
            _nlp_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-small", local_files_only=True
            )
        except Exception:
            pass
    return _nlp_model, _nlp_tokenizer


def _parse_vitals(context: str | None) -> dict[str, str | float | None]:
    """Extract numeric vitals from the context string produced by the API."""
    out: dict[str, str | float | None] = {}
    if not context:
        return out
    patterns = {
        "heart_rate": r"heart_rate=([\d.]+)",
        "bp_systolic": r"bp=([\d.]+)/",
        "bp_diastolic": r"bp=[\d.]+/([\d.]+)",
        "spo2": r"spo2=([\d.]+)",
        "body_temp_c": r"body_temp_c=([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, context)
        out[key] = float(m.group(1)) if m else None

    alerts_match = re.search(r"alerts:\s*(.+)$", context)
    out["alerts_raw"] = alerts_match.group(1).strip() if alerts_match else ""
    return out


def _assess_vital(name: str, value: float | None) -> tuple[str, str]:
    """Return (status, explanation) for a single vital."""
    if value is None:
        return "unavailable", "No data recorded."

    ranges = {
        "heart_rate": [
            (60, 100, "normal", "Your heart rate of {v:.0f} bpm is within the normal resting range (60–100 bpm)."),
            (50, 60, "low-normal", "Your heart rate of {v:.0f} bpm is on the low side. This can be normal for fit individuals but may warrant monitoring."),
            (100, 120, "elevated", "Your heart rate of {v:.0f} bpm is elevated. This can occur with exertion, stress, or caffeine. Persistent tachycardia should be evaluated."),
            (0, 50, "low", "Your heart rate of {v:.0f} bpm is below normal. Bradycardia may need clinical evaluation if symptomatic."),
            (120, 300, "high", "Your heart rate of {v:.0f} bpm is significantly elevated. Seek clinical evaluation if persistent."),
        ],
        "spo2": [
            (95, 101, "normal", "Oxygen saturation of {v:.0f}% is within the healthy range (95–100%)."),
            (92, 95, "borderline", "Oxygen saturation of {v:.0f}% is slightly below optimal. Monitor closely and ensure the sensor is positioned correctly."),
            (88, 92, "low", "Oxygen saturation of {v:.0f}% is below normal. This may indicate respiratory compromise — clinical review is recommended."),
            (0, 88, "critical", "Oxygen saturation of {v:.0f}% is critically low. Immediate medical attention is needed."),
        ],
        "body_temp_c": [
            (36.1, 37.5, "normal", "Body temperature of {v:.1f}°C is within the normal range (36.1–37.5°C)."),
            (37.5, 38.0, "mild", "Body temperature of {v:.1f}°C is mildly elevated. This could indicate early fever — monitor for changes."),
            (38.0, 39.5, "fever", "Body temperature of {v:.1f}°C indicates a fever. Consider hydration, rest, and clinical evaluation if it persists."),
            (39.5, 45, "high_fever", "Body temperature of {v:.1f}°C is a high fever. Medical evaluation is strongly recommended."),
            (30, 36.1, "low", "Body temperature of {v:.1f}°C is below normal range. Hypothermia risk should be assessed."),
        ],
        "bp_systolic": [
            (90, 120, "normal", "Systolic blood pressure of {v:.0f} mmHg is within the normal range."),
            (120, 130, "elevated", "Systolic blood pressure of {v:.0f} mmHg is slightly elevated. Lifestyle factors like salt intake and stress may contribute."),
            (130, 140, "high_stage1", "Systolic blood pressure of {v:.0f} mmHg falls in the Stage 1 hypertension range. Regular monitoring and lifestyle changes are advised."),
            (140, 180, "high_stage2", "Systolic blood pressure of {v:.0f} mmHg is in the Stage 2 hypertension range. Clinical follow-up is recommended."),
            (180, 300, "crisis", "Systolic blood pressure of {v:.0f} mmHg is dangerously high. Seek immediate medical attention."),
            (0, 90, "low", "Systolic blood pressure of {v:.0f} mmHg is low. Hypotension can cause dizziness — consult a clinician if symptomatic."),
        ],
    }
    vital_ranges = ranges.get(name, [])
    for low, high, status, explanation in vital_ranges:
        if low <= value < high:
            return status, explanation.format(v=value)
    return "unknown", f"{name} value of {value} could not be assessed."


def _build_status_summary(vitals: dict) -> str:
    """Compose a full patient status overview from parsed vitals."""
    lines = ["Here's a summary of the current demo readings:\n"]
    vital_labels = {
        "heart_rate": "Heart Rate",
        "spo2": "Oxygen Saturation (SpO2)",
        "body_temp_c": "Body Temperature",
        "bp_systolic": "Blood Pressure (Systolic)",
    }
    concerns = []
    for key, label in vital_labels.items():
        val = vitals.get(key)
        status, explanation = _assess_vital(key, val)
        lines.append(f"**{label}:** {explanation}")
        if status not in ("normal", "low-normal", "unavailable"):
            concerns.append(label)

    bp_dia = vitals.get("bp_diastolic")
    if bp_dia is not None:
        if bp_dia >= 90:
            lines.append(f"**Blood Pressure (Diastolic):** {bp_dia:.0f} mmHg is elevated (normal is below 80–90 mmHg).")
            concerns.append("Blood Pressure (Diastolic)")
        elif bp_dia < 60:
            lines.append(f"**Blood Pressure (Diastolic):** {bp_dia:.0f} mmHg is low — monitor for symptoms like dizziness.")
        else:
            lines.append(f"**Blood Pressure (Diastolic):** {bp_dia:.0f} mmHg is within normal range.")

    alerts_raw = vitals.get("alerts_raw", "")
    if alerts_raw and alerts_raw != "no active alerts":
        lines.append(f"\n**Active Alerts:** {alerts_raw}")

    if concerns:
        lines.append(f"\n**Areas to watch:** {', '.join(concerns)}. Consider discussing these with a healthcare provider.")
    else:
        lines.append("\nAll readings appear within normal ranges for this demo session.")
    return "\n".join(lines)


def _contextual_response(message: str, context: str | None) -> str:
    """Generate an intelligent response by matching intent and using vitals data."""
    text = message.lower().strip()
    vitals = _parse_vitals(context)
    has_vitals = any(vitals.get(k) is not None for k in ("heart_rate", "spo2", "body_temp_c", "bp_systolic"))

    if any(w in text for w in ("emergency", "chest pain", "can't breathe", "cannot breathe", "unconscious", "bleeding", "suicide")):
        return (
            "If you or someone else may be experiencing an emergency, please call your local emergency number "
            "immediately (e.g. 112, 911, 199). This application is a demo and cannot provide emergency assistance."
        )

    if any(w in text for w in ("status", "how am i", "overview", "summary", "readings", "vitals", "results", "report")):
        if has_vitals:
            return _build_status_summary(vitals)
        return "No vitals data is available yet. Try simulating or adding readings first, then ask again."

    if "heart" in text or "pulse" in text or "bpm" in text or "heart rate" in text:
        hr = vitals.get("heart_rate")
        _, explanation = _assess_vital("heart_rate", hr)
        extra = (
            " A normal resting heart rate for adults is typically 60–100 bpm. "
            "Factors like fitness level, stress, caffeine, and medications can all influence heart rate."
        )
        return explanation + extra

    if "oxygen" in text or "spo2" in text or "o2" in text or "saturation" in text:
        spo2 = vitals.get("spo2")
        _, explanation = _assess_vital("spo2", spo2)
        extra = (
            " Healthy SpO2 is generally 95–100%. Readings can be affected by cold fingers, nail polish, or "
            "poor sensor contact. Persistently low values warrant medical evaluation."
        )
        return explanation + extra

    if "blood pressure" in text or "bp " in text or "hypertension" in text or "systolic" in text or "diastolic" in text:
        sys_val = vitals.get("bp_systolic")
        dia_val = vitals.get("bp_diastolic")
        _, sys_expl = _assess_vital("bp_systolic", sys_val)
        parts = [sys_expl]
        if dia_val is not None:
            if dia_val >= 90:
                parts.append(f"Diastolic pressure of {dia_val:.0f} mmHg is also elevated.")
            else:
                parts.append(f"Diastolic pressure of {dia_val:.0f} mmHg is within normal range.")
        parts.append(
            "Normal blood pressure is generally below 120/80 mmHg. Regular monitoring, "
            "reduced sodium intake, exercise, and stress management can all help maintain healthy levels."
        )
        return " ".join(parts)

    if "temperature" in text or "temp" in text or "fever" in text:
        temp = vitals.get("body_temp_c")
        _, explanation = _assess_vital("body_temp_c", temp)
        extra = (
            " Normal body temperature ranges from about 36.1°C to 37.5°C. "
            "It can fluctuate with time of day, activity, and environment."
        )
        return explanation + extra

    if "medication" in text or "drug" in text or "dose" in text or "medicine" in text:
        return (
            "Medication decisions should always be made with a qualified healthcare professional. "
            "This demo cannot recommend, adjust, or evaluate medications. If you have questions about "
            "your prescriptions, please consult your doctor or pharmacist."
        )

    if any(w in text for w in ("risk", "score", "prediction", "ai analysis", "alert")):
        if has_vitals:
            alerts_raw = vitals.get("alerts_raw", "")
            if alerts_raw and alerts_raw != "no active alerts":
                return (
                    f"Based on current readings, the system has generated the following alerts: {alerts_raw}. "
                    "The AI risk score combines multiple vital sign trends to estimate overall health risk. "
                    "Check the AI Analysis page for a detailed breakdown including ML model predictions."
                )
            return (
                "Currently there are no active alerts. The AI risk engine evaluates heart rate, blood pressure, "
                "SpO2, and temperature trends to calculate a composite risk score. You can view detailed predictions "
                "on the AI Analysis page."
            )
        return "No vitals data available yet. Add or simulate readings to generate risk predictions."

    if any(w in text for w in ("hello", "hi ", "hi!", "hey", "good morning", "good afternoon", "good evening")):
        if has_vitals:
            hr = vitals.get("heart_rate")
            spo2 = vitals.get("spo2")
            hr_str = f"HR {hr:.0f} bpm" if hr else ""
            spo2_str = f"SpO2 {spo2:.0f}%" if spo2 else ""
            brief = ", ".join(filter(None, [hr_str, spo2_str]))
            return (
                f"Hello! I'm monitoring the patient's vitals. Current quick snapshot: {brief}. "
                "Feel free to ask about any specific vital sign, the overall status, or risk predictions."
            )
        return "Hello! I'm the HealthPulse AI assistant. Ask me about patient vitals, risk scores, or general health topics."

    if any(w in text for w in ("what can you do", "help", "capabilities", "features", "how do you work")):
        return (
            "I can help you with:\n\n"
            "- **Patient Status** — Ask 'How am I doing?' or 'Show my status' for a full vitals summary\n"
            "- **Specific Vitals** — Ask about heart rate, blood pressure, SpO2, or temperature\n"
            "- **Risk Analysis** — Ask about risk scores or active alerts\n"
            "- **Health Information** — General wellness questions and health literacy\n\n"
            "Try asking something like 'What's my heart rate?' or 'Give me an overview.'"
        )

    if any(w in text for w in ("sleep", "exercise", "diet", "nutrition", "stress", "anxiety", "weight")):
        topic_responses = {
            "sleep": (
                "Quality sleep is essential for health. Most adults need 7–9 hours per night. "
                "Poor sleep can affect heart rate, blood pressure, and overall wellbeing. "
                "Maintaining a consistent sleep schedule and limiting screen time before bed can help."
            ),
            "exercise": (
                "Regular physical activity strengthens the cardiovascular system and can improve heart rate, "
                "blood pressure, and oxygen utilisation. The general recommendation is at least 150 minutes "
                "of moderate aerobic activity per week. Always consult a clinician before starting a new program."
            ),
            "diet": (
                "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins supports "
                "cardiovascular health. Reducing sodium can help manage blood pressure, and staying hydrated "
                "supports overall body function."
            ),
            "nutrition": (
                "Good nutrition plays a key role in maintaining healthy vitals. Focus on whole foods, "
                "adequate hydration, and balanced macronutrients. Specific dietary needs should be discussed "
                "with a healthcare professional or registered dietitian."
            ),
            "stress": (
                "Chronic stress can elevate heart rate and blood pressure. Techniques like deep breathing, "
                "mindfulness, regular exercise, and adequate sleep can help manage stress levels. "
                "If stress is significantly impacting your life, consider speaking with a professional."
            ),
            "anxiety": (
                "Anxiety can manifest physically with increased heart rate, elevated blood pressure, and "
                "changes in breathing patterns. If you're experiencing persistent anxiety, cognitive behavioral "
                "therapy and other treatments can be very effective — please consult a healthcare provider."
            ),
            "weight": (
                "Maintaining a healthy weight supports cardiovascular health and can improve blood pressure, "
                "heart rate, and overall energy levels. Sustainable changes in diet and activity are more "
                "effective than extreme measures. A healthcare professional can help set realistic goals."
            ),
        }
        for topic, response in topic_responses.items():
            if topic in text:
                return response

    if has_vitals:
        summary = _build_status_summary(vitals)
        return (
            f"I'd be happy to help! Here's what I can tell you based on the current readings:\n\n"
            f"{summary}\n\n"
            "Feel free to ask about specific vitals, risk scores, or general health topics."
        )

    return (
        "I can help with patient monitoring insights and general health information. "
        "Try asking about specific vitals (e.g. 'What's my heart rate?'), "
        "your overall status, or general wellness topics like sleep, exercise, or stress management."
    )


def generate_reply(user_message: str, context: str | None = None) -> tuple[str, bool]:
    # 0. Try Groq if configured (prioritized as per user request)
    if settings.groq_configured:
        try:
            # Groq is fully compatible with the OpenAI python client!
            client = openai.OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            system_prompt = (
                "You are HealthPulse AI, a virtual health assistant in a practitioner portal. "
                "Provide concise, safe, and helpful answers based on the patient's vitals context. "
                "You are talking to a healthcare practitioner viewing the patient's dashboard, or a simulated patient. "
                "Keep answers brief (1-3 sentences) and readable. Use markdown formatting."
            )
            
            prompt_content = f"Patient Context: {context}\n" if context else ""
            prompt_content += f"User Question: {user_message}"
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content}
                ],
                max_tokens=250,
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            return content + DISCLAIMER, True
        except Exception as e:
            err_msg = str(e)
            print(f"Groq API error: {err_msg}")
            # Fall through to the next fallback

    # 1. Try OpenAI if a key is configured (and Groq wasn't configured)
    if settings.openai_configured:
        try:
            client = openai.OpenAI(api_key=settings.openai_api_key)
            system_prompt = (
                "You are HealthPulse AI, a virtual health assistant in a practitioner portal. "
                "Provide concise, safe, and helpful answers based on the patient's vitals context. "
                "You are talking to a healthcare practitioner viewing the patient's dashboard, or a simulated patient. "
                "Keep answers brief (1-3 sentences) and readable. Use markdown formatting."
            )
            
            prompt_content = f"Patient Context: {context}\n" if context else ""
            prompt_content += f"User Question: {user_message}"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content}
                ],
                max_tokens=250,
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            return content + DISCLAIMER, True
        except Exception as e:
            err_msg = str(e)
            print(f"OpenAI API error: {err_msg}")
            # Fall through to the next fallback

    # 2. Fall back to local Flan-T5 model
    model, tokenizer = _get_nlp_model()
    if model is not None and tokenizer is not None:
        try:
            prompt = (
                "You are a helpful health information assistant in a software demo. "
                "Give a concise, safe, general wellness answer.\n"
            )
            if context:
                prompt += f"Patient Context: {context}\n"
            prompt += f"Question: {user_message}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            content = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if len(content) >= 10:
                content = re.sub(r"^```\w*\n|```$", "", content, flags=re.MULTILINE)
                return content + DISCLAIMER, True
        except Exception:
            pass

    # 3. Final fallback to keyword rules
    return _contextual_response(user_message, context), False
