"""
HealthPulse AI — Virtual Health Monitoring MVP
Requires: streamlit, pandas, openai
  pip install streamlit pandas openai
"""

import random
from datetime import datetime
import os

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------------------------
# Configuration — edit thresholds here, not buried in logic
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "heart_rate_high": 100,       # bpm
    "temperature_high": 38.0,     # °C
    "oxygen_low": 95,             # %
    "systolic_high": 140,         # mmHg
    "alert_consecutive": 2,       # readings required before alert fires
}

RECOMMENDATIONS = {
    "High Heart Rate": "Rest for 5–10 minutes and monitor condition.",
    "Fever": "Hydrate and monitor temperature regularly.",
    "Low Oxygen": "Take deep breaths and seek medical advice if persistent.",
    "High Blood Pressure": "Reduce activity, relax, and recheck blood pressure.",
}


# ---------------------------------------------------------------------------
# Environment / API key helpers
# ---------------------------------------------------------------------------

def _load_env_file():
    """
    Load .env into os.environ for local development only.
    Skipped automatically on Streamlit Cloud / hosted environments.
    """
    if os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("IS_CLOUD_DEPLOY"):
        return
    env_path = ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    return key if key else None


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(previous=None):
    """Generate realistic vital signs with slight drift over time."""
    if previous is None:
        heart_rate  = random.randint(72, 88)
        temperature = round(random.uniform(36.4, 37.2), 1)
        oxygen_level = random.randint(96, 99)
        systolic    = random.randint(112, 126)
        diastolic   = random.randint(72, 84)
    else:
        heart_rate   = previous["heart_rate"]   + random.randint(-3, 3)
        temperature  = round(previous["temperature"] + random.uniform(-0.2, 0.2), 1)
        oxygen_level = previous["oxygen_level"] + random.randint(-1, 1)
        systolic     = previous["systolic"]     + random.randint(-4, 4)
        diastolic    = previous["diastolic"]    + random.randint(-3, 3)

    heart_rate   = max(60,  min(120, heart_rate))
    temperature  = max(36.0, min(39.0, temperature))
    oxygen_level = max(90,  min(100, oxygen_level))
    systolic     = max(95,  min(180, systolic))
    diastolic    = max(55,  min(110, diastolic))

    return {
        "heart_rate":    heart_rate,
        "temperature":   temperature,
        "oxygen_level":  oxygen_level,
        "systolic":      systolic,
        "diastolic":     diastolic,
        "blood_pressure": f"{systolic}/{diastolic}",
        "timestamp":     datetime.now(),
    }


# ---------------------------------------------------------------------------
# Health analysis — hysteresis prevents flickering alerts
# ---------------------------------------------------------------------------

def analyze_health(vitals, consecutive_counts: dict):
    """
    Raise an alert only after THRESHOLDS['alert_consecutive'] back-to-back
    out-of-range readings for that metric. Returns (active_issues, updated_counts).
    """
    checks = {
        "High Heart Rate":    vitals["heart_rate"]   > THRESHOLDS["heart_rate_high"],
        "Fever":              vitals["temperature"]  > THRESHOLDS["temperature_high"],
        "Low Oxygen":         vitals["oxygen_level"] < THRESHOLDS["oxygen_low"],
        "High Blood Pressure": vitals["systolic"]    > THRESHOLDS["systolic_high"],
    }

    required = THRESHOLDS["alert_consecutive"]
    updated_counts = {}
    active_issues = []

    for issue, triggered in checks.items():
        count = consecutive_counts.get(issue, 0)
        updated_counts[issue] = count + 1 if triggered else 0
        if updated_counts[issue] >= required:
            active_issues.append(issue)

    return active_issues, updated_counts


def get_alerts(issues):
    return [
        {"issue": i, "recommendation": RECOMMENDATIONS.get(i, "Monitor condition.")}
        for i in issues
    ]


# ---------------------------------------------------------------------------
# Chatbot — multi-turn history passed to OpenAI
# ---------------------------------------------------------------------------

def chatbot_response(user_message: str, vitals=None, issues=None, chat_history=None):
    """
    Send the full conversation history to OpenAI for coherent multi-turn replies.
    Falls back to rule-based responses when no API key is set or the call fails.
    """
    api_key = _get_openai_api_key()

    if api_key and OpenAI is not None:
        status_context = ", ".join(issues) if issues else "stable"
        vitals_context = (
            f"heart_rate={vitals.get('heart_rate')} bpm, "
            f"temperature={vitals.get('temperature')} °C, "
            f"oxygen_level={vitals.get('oxygen_level')}%, "
            f"blood_pressure={vitals.get('blood_pressure')} mmHg"
        ) if vitals else "No vitals available."

        # System context.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise virtual health monitoring assistant for a prototype app. "
                    "Do not diagnose. Give practical, safe, non-emergency guidance in 2–4 sentences."
                ),
            },
            {
                "role": "system",
                "content": f"Current patient status: {status_context}. Vitals: {vitals_context}",
            },
        ]

        # Replay prior conversation turns so the model has context.
        for role_label, text in (chat_history or []):
            role = "user" if role_label == "You" else "assistant"
            messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": user_message})

        try:
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=150,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            pass  # Fall through to local rules.

    # ---- Local rule-based fallback ----
    msg = user_message.lower().strip()
    if "dizzy" in msg or "weak" in msg:
        return "You may be unstable. Please check your vitals now and sit down safely."
    if "high bp" in msg or "blood pressure" in msg:
        return "High blood pressure means your force of blood flow is above normal. Rest and re-check after 5 minutes."
    if "heart rate" in msg or "pulse" in msg:
        return "A normal resting heart rate is 60–100 bpm. Higher values may indicate stress, activity, or illness."
    if "oxygen" in msg:
        return "Oxygen level below 95% can be concerning. Re-check and seek help if it stays low."
    if "status" in msg or "how am i" in msg:
        if vitals and issues is not None:
            if issues:
                return f"Current status: {', '.join(issues)} detected. Please follow alert recommendations."
            return "Current status looks stable. Keep monitoring regularly."
    return "I can help with symptoms and vitals. Try: 'I feel dizzy', 'What is high BP?', or 'How am I?'."


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HealthPulse AI - MVP", layout="wide")
_load_env_file()

st.title("HealthPulse AI — Virtual Health Monitoring")
st.caption("Real-time AI-powered patient monitoring prototype")

# Basic styling to make the left panel closer to the requested mockup.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 270px;
        max-width: 270px;
        border-right: 1px solid #e8e8ef;
    }
    .hp-brand {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .hp-subbrand {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 18px;
    }
    .hp-section-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #9ca3af;
        margin: 12px 0 8px 0;
    }
    .hp-nav-item {
        padding: 10px 12px;
        border-radius: 10px;
        margin: 4px 0;
        font-size: 14px;
        background: #f8fafc;
    }
    .hp-nav-active {
        background: #e8f0ff;
        border-left: 3px solid #2f80ed;
        font-weight: 600;
    }
    .hp-footer {
        margin-top: 18px;
        font-size: 14px;
        color: #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session state initialisation ----
defaults = {
    "vitals":            None,
    "heart_rate_history": [],
    "chat_history":      [],
    "consecutive_counts": {},
    "issues":            [],
    "last_vitals_update": datetime.now(),
    "paused":            False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.vitals is None:
    st.session_state.vitals = generate_data()
    st.session_state.heart_rate_history = [st.session_state.vitals["heart_rate"]]
if not st.session_state.issues:
    st.session_state.issues, st.session_state.consecutive_counts = analyze_health(
        st.session_state.vitals, st.session_state.consecutive_counts
    )

# ---- Sidebar controls ----
with st.sidebar:
    st.markdown('<div class="hp-brand">HealthPulse AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hp-subbrand">Central Medical<br/>PRACTITIONER PORTAL</div>', unsafe_allow_html=True)

    st.markdown('<div class="hp-section-title">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hp-nav-item hp-nav-active">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hp-nav-item">🗂️ Patient Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="hp-nav-item">🤖 AI Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hp-nav-item">🚨 Clinical Alerts</div>', unsafe_allow_html=True)
    st.markdown('<div class="hp-nav-item">📈 Reports</div>', unsafe_allow_html=True)

    st.markdown('<div class="hp-section-title">Monitoring</div>', unsafe_allow_html=True)
    update_interval = st.slider(
        "Update interval (seconds)", min_value=2, max_value=8, value=3, step=1
    )
    st.session_state.paused = st.toggle(
        "⏸ Pause monitoring", value=st.session_state.paused
    )
    st.markdown("---")
    if bool(_get_openai_api_key()):
        st.success("✅ OpenAI API connected")
    else:
        st.info("ℹ️ Local fallback mode\n\nSet `OPENAI_API_KEY` for LLM chat.")

    st.markdown("---")
    st.markdown('<div class="hp-footer">🛟 Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="hp-footer">🚪 Log Out</div>', unsafe_allow_html=True)

@st.fragment(run_every=f"{update_interval}s")
def monitoring_tick():
    """Periodic non-blocking vitals update + health analysis."""
    if st.session_state.paused:
        return

    now = datetime.now()
    elapsed = (now - st.session_state.last_vitals_update).total_seconds()
    if elapsed < update_interval:
        return

    st.session_state.vitals = generate_data(st.session_state.vitals)
    st.session_state.last_vitals_update = now
    st.session_state.heart_rate_history.append(st.session_state.vitals["heart_rate"])
    st.session_state.heart_rate_history = st.session_state.heart_rate_history[-20:]
    st.session_state.issues, st.session_state.consecutive_counts = analyze_health(
        st.session_state.vitals, st.session_state.consecutive_counts
    )


monitoring_tick()

current_vitals = st.session_state.vitals
issues = st.session_state.issues
alerts = get_alerts(issues)

# ---- Status bar ----
last_updated_str = current_vitals["timestamp"].strftime("%H:%M:%S")
status_label = "⏸ Paused" if st.session_state.paused else "🟢 Live"
st.caption(f"{status_label}  ·  Last updated: {last_updated_str}")

# ---- Vitals cards ----
c1, c2, c3, c4 = st.columns(4, gap="small")
c1.metric("Heart Rate",    f"{current_vitals['heart_rate']} bpm")
c2.metric("Temperature",   f"{current_vitals['temperature']:.1f} °C")
c3.metric("Oxygen Level",  f"{current_vitals['oxygen_level']} %")
c4.metric("Blood Pressure", f"{current_vitals['blood_pressure']} mmHg")

st.markdown("---")

# ---- Clinical alerts ----
st.subheader("Clinical Alerts")
if alerts:
    for alert in alerts:
        st.error(f"**{alert['issue']}** — {alert['recommendation']}")
else:
    st.success("No abnormal conditions detected. Vitals are within expected range.")

st.markdown("---")

# ---- Heart-rate chart + chatbot ----
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.subheader("Heart Rate Trend")
    chart_df = pd.DataFrame({
        "Reading":    list(range(1, len(st.session_state.heart_rate_history) + 1)),
        "Heart Rate": st.session_state.heart_rate_history,
    })
    st.line_chart(chart_df.set_index("Reading"))

with right_col:
    st.subheader("HealthPulse AI Assistant")
    chat_box = st.container(height=320, border=True)
    with chat_box:
        if not st.session_state.chat_history:
            st.write("**Assistant:** Ask about symptoms or vitals to get guidance.")
        for role, text in st.session_state.chat_history:
            st.write(f"**{role}:** {text}")

    user_input = st.text_input(
        "Type your health question", placeholder="e.g. I feel dizzy"
    )
    if st.button("Send", use_container_width=True) and user_input.strip():
        question = user_input.strip()
        # Snapshot history *before* appending so the model doesn't see the
        # current message twice (it's passed separately as the final user turn).
        prior_history = list(st.session_state.chat_history)
        st.session_state.chat_history.append(("You", question))
        reply = chatbot_response(
            question,
            vitals=current_vitals,
            issues=issues,
            chat_history=prior_history,
        )
        st.session_state.chat_history.append(("Assistant", reply))
        st.rerun()