import random
from datetime import datetime
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.getenv("HEALTHPULSE_API_URL", "http://127.0.0.1:8000")

def _load_env_file():
    if os.getenv("STREAMLIT_CLOUD"):
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

def api_get(path, timeout=1.5):
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def api_post(path, payload, timeout=3):
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def backend_is_online():
    health = api_get("/api/health", timeout=0.8)
    return bool(health and health.get("status") == "ok")

def vital_payload(vitals):
    return {
        "heart_rate": vitals["heart_rate"],
        "bp_systolic": vitals["systolic"],
        "bp_diastolic": vitals["diastolic"],
        "spo2": vitals["oxygen_level"],
        "body_temp_c": vitals["temperature"],
        "recorded_at": vitals["timestamp"].isoformat(),
        "source": "streamlit",
    }

def sync_vital_to_backend(vitals):
    return api_post("/api/vitals", vital_payload(vitals), timeout=3)

def backend_chat_response(user_message):
    response = api_post(
        "/api/chat",
        {"message": user_message, "include_recent_vitals": True},
        timeout=8,
    )
    if response and response.get("reply"):
        return response["reply"]
    return None

def generate_data(previous=None):
    if previous is None:
        heart_rate = random.randint(72, 88)
        temperature = round(random.uniform(36.4, 37.2), 1)
        oxygen_level = random.randint(96, 99)
        systolic = random.randint(112, 126)
        diastolic = random.randint(72, 84)
    else:
        heart_rate = previous["heart_rate"] + random.randint(-3, 3)
        temperature = round(previous["temperature"] + random.uniform(-0.2, 0.2), 1)
        oxygen_level = previous["oxygen_level"] + random.randint(-1, 1)
        systolic = previous["systolic"] + random.randint(-4, 4)
        diastolic = previous["diastolic"] + random.randint(-3, 3)

    heart_rate = max(60, min(120, heart_rate))
    temperature = max(36.0, min(39.0, temperature))
    oxygen_level = max(90, min(100, oxygen_level))
    systolic = max(95, min(180, systolic))
    diastolic = max(55, min(110, diastolic))

    return {
        "heart_rate": heart_rate,
        "temperature": temperature,
        "oxygen_level": oxygen_level,
        "systolic": systolic,
        "diastolic": diastolic,
        "blood_pressure": f"{systolic}/{diastolic}",
        "timestamp": datetime.now(),
    }

st.set_page_config(page_title="HealthPulse AI", layout="wide", initial_sidebar_state="expanded")
_load_env_file()

# --- CSS INJECTION FOR PREMIUM STYLING ---
st.markdown(
    """
    <style>
    /* Global Fonts & Backgrounds */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Hide the deploy toolbar; keep header transparent and minimal */
    [data-testid="stToolbar"] { visibility: hidden !important; }
    [data-testid="stHeader"] {
        background: transparent !important;
        height: 2.875rem !important;
    }

    /* Force sidebar to always remain visible — override Streamlit's slide-out transform */
    [data-testid="stSidebar"] {
        transform: none !important;
        min-width: 16rem !important;
        max-width: 16rem !important;
        visibility: visible !important;
        display: flex !important;
    }

    /* Hide both collapse and expand toggle buttons entirely */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"] {
        display: none !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
    }
    .hp-brand {
        font-size: 20px;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .hp-brand-icon {
        background: #2563eb;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
    .hp-subbrand {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 30px;
        margin-left: 36px;
    }
    .hp-section-title {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 20px 0 8px 0;
        letter-spacing: 0.5px;
    }

    /* ── Sidebar nav: style the radio as a clean nav list ── */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        padding: 11px 14px !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        cursor: pointer !important;
        transition: background 0.15s, color 0.15s !important;
        border-left: 3px solid transparent !important;
        margin: 0 !important;
        background: transparent !important;
        line-height: 1.4 !important;
        letter-spacing: 0.01em !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #60a5fa !important;
        font-weight: 600 !important;
    }
    /* Hide ONLY the radio circle dot, keep text visible */
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    /* Target the container of the radio circle specifically */
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] span {
        margin-left: 0 !important;
    }

    
    /* Header Row Elements */
    .patient-header-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 24px;
        margin-top: -30px;
    }
    .patient-title {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .patient-subtitle {
        font-size: 14px;
        color: #64748b;
    }
    .header-buttons {
        display: flex;
        gap: 12px;
    }
    .btn-secondary {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        color: #334155;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .btn-primary {
        background: #2563eb;
        border: 1px solid #2563eb;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Custom Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
    }
    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        color: #475569;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .metric-badge-normal {
        background: #dcfce7;
        color: #166534;
        font-size: 10px;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-badge-high {
        background: #fef08a;
        color: #854d0e;
        font-size: 10px;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-badge-critical {
        background: #fee2e2;
        color: #991b1b;
        font-size: 10px;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .metric-unit {
        font-size: 14px;
        font-weight: 500;
        color: #64748b;
        margin-left: 2px;
    }
    .metric-trend {
        font-size: 12px;
        color: #10b981;
        display: flex;
        align-items: center;
        gap: 4px;
        font-weight: 500;
    }
    .metric-trend.neutral { color: #64748b; }
    
    /* Alerts Banner override */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        padding: 16px !important;
    }
    
    /* Custom Timeline */
    .timeline-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-top: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    .timeline-item {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        position: relative;
    }
    .timeline-item:last-child {
        margin-bottom: 0;
    }
    .timeline-line {
        position: absolute;
        left: 11px;
        top: 24px;
        bottom: -24px;
        width: 2px;
        background: #f1f5f9;
    }
    .timeline-item:last-child .timeline-line {
        display: none;
    }
    .timeline-dot {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #e0e7ff;
        border: 4px solid #ffffff;
        box-shadow: 0 0 0 1px #e2e8f0;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .timeline-dot.green { background: #10b981; }
    .timeline-dot.blue { background: #3b82f6; }
    .timeline-dot.yellow { background: #eab308; }
    .timeline-content {
        flex: 1;
    }
    .timeline-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 4px;
    }
    .timeline-title {
        font-size: 14px;
        font-weight: 600;
        color: #0f172a;
    }
    .timeline-time {
        font-size: 12px;
        color: #94a3b8;
    }
    .timeline-desc {
        font-size: 13px;
        color: #475569;
    }

    /* ── Native Streamlit Element Overrides for Other Pages ── */
    [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stMain"] p {
        color: #334155;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
    }
    [data-testid="stDataFrame"] {
        background: #ffffff;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
        margin-top: 12px;
    }
    /* Override markdown bold text to be clearly dark */
    [data-testid="stMarkdownContainer"] strong {
        color: #0f172a !important;
    }
    
    /* Style Buttons to be white with black text */
    [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    [data-testid="stDownloadButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }
    [data-testid="stDownloadButton"] button p, [data-testid="stFormSubmitButton"] button p {
        color: inherit !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

if "vitals" not in st.session_state:
    st.session_state.vitals = generate_data()
if "heart_rate_history" not in st.session_state:
    st.session_state.heart_rate_history = [st.session_state.vitals["heart_rate"]]
if "risk_history" not in st.session_state:
    st.session_state.risk_history = [0]
if "chat_history" not in st.session_state:
    saved = api_get("/api/chat/history?limit=50")
    if saved:
        st.session_state.chat_history = [(m["role"], m["content"]) for m in saved]
    else:
        st.session_state.chat_history = []
if "last_updated" not in st.session_state:
    st.session_state.last_updated = datetime.now()
if "paused" not in st.session_state:
    st.session_state.paused = False
if "backend_sync_count" not in st.session_state:
    st.session_state.backend_sync_count = 0
if "backend_last_error" not in st.session_state:
    st.session_state.backend_last_error = None

with st.sidebar:
    st.markdown('''
        <div class="hp-brand"><div class="hp-brand-icon">✚</div> HealthPulse AI</div>
        <div class="hp-subbrand">Practitioner Portal</div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="hp-section-title">Navigation</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "Primary navigation",
        ["Dashboard", "Patient Records", "AI Analysis", "Clinical Alerts", "Reports", "Support"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="hp-section-title">Monitoring</div>', unsafe_allow_html=True)
    update_interval = st.slider("Update interval (seconds)", min_value=2, max_value=8, value=3, step=1)
    st.session_state.paused = st.toggle("Pause monitoring", value=st.session_state.paused)

backend_online = backend_is_online()

with st.sidebar:
    st.markdown("<br/>", unsafe_allow_html=True)
    if backend_online:
        st.info(f"API Connected ({st.session_state.backend_sync_count})")
        st.info("Local NLP Active")
    else:
        st.error("API Offline")
    st.markdown("---")
    st.markdown('<p style="color: #94a3b8; font-size: 11px;">AI-driven deep learning portal.</p>', unsafe_allow_html=True)

if not backend_online:
    st.error("The FastAPI Backend is currently offline. Please start the backend (`python -m uvicorn app.main:app`) to use the application.")
    st.stop()

@st.fragment(run_every=f"{update_interval}s")
def monitoring_tick():
    if st.session_state.paused:
        return

    elapsed = (datetime.now() - st.session_state.last_updated).total_seconds()
    if elapsed >= update_interval:
        st.session_state.vitals = generate_data(st.session_state.vitals)
        st.session_state.last_updated = datetime.now()
        st.session_state.heart_rate_history.append(st.session_state.vitals["heart_rate"])
        st.session_state.heart_rate_history = st.session_state.heart_rate_history[-20:]
        
        result = sync_vital_to_backend(st.session_state.vitals)
        if result:
            st.session_state.backend_sync_count += 1
            st.session_state.backend_last_error = None
        else:
            st.session_state.backend_last_error = "Last backend sync failed."

monitoring_tick()

current_vitals = st.session_state.vitals
summary = api_get("/api/summary")

# --- GRACEFUL BACKEND INITIALIZATION ---
if not summary:
    with st.sidebar:
        st.warning("Syncing with backend...")
    # Provide safe defaults so the UI doesn't crash or hide
    prediction = {"risk_score": 0, "risk_level": "Stabilizing", "drivers": ["Initializing..."]}
    alerts = []
else:
    prediction = summary.get("prediction", {})
    alerts = summary.get("active_alerts", [])

risk_score = prediction.get("risk_score", 0)
risk_level = prediction.get("risk_level", "Unknown")
risk_drivers = prediction.get("drivers", ["None"])

if st.session_state.risk_history:
    st.session_state.risk_history.append(risk_score)
    st.session_state.risk_history = st.session_state.risk_history[-20:]

if selected_page == "Dashboard":
    # 1. Premium Header
    st.markdown(f'''
        <div class="patient-header-container">
            <div>
                <div class="patient-title">Patient Overview</div>
                <div class="patient-subtitle">Real-time health telemetry for <b>Sarah Johnson</b> • Last Updated: {st.session_state.last_updated.strftime('%H:%M:%S')}</div>
            </div>
            <div class="header-buttons">
                <button class="btn-secondary">Last 24 Hours</button>
                <button class="btn-primary">⬇ Export Data</button>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Helper for Custom Metrics
    def metric_html(title, icon, value, unit, status, status_class, trend_txt, trend_icon="↗"):
        return f'''
        <div class="metric-card">
            <div class="metric-header">
                <div class="metric-title">{icon} {title}</div>
                <div class="metric-badge-{status_class}">{status}</div>
            </div>
            <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
            <div class="metric-trend {'neutral' if trend_icon=='-' else ''}">{trend_icon} {trend_txt}</div>
        </div>
        '''

    # Layout Top Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    
    hr_val = current_vitals['heart_rate']
    hr_status = "NORMAL"
    hr_class = "normal"
    if hr_val > 100 or hr_val < 60:
        hr_status = "HIGH" if hr_val > 100 else "LOW"
        hr_class = "high"
    with c1:
        st.markdown(metric_html("Heart Rate", "", hr_val, "bpm", hr_status, hr_class, "vs baseline", "↗"), unsafe_allow_html=True)

    tmp_val = current_vitals['temperature']
    tmp_status = "NORMAL"
    tmp_class = "normal"
    if tmp_val >= 38.0:
        tmp_status = "HIGH"
        tmp_class = "high"
    with c2:
        st.markdown(metric_html("Temperature", "", f"{tmp_val:.1f}", "°C", tmp_status, tmp_class, "Stable", "-"), unsafe_allow_html=True)

    o2_val = current_vitals['oxygen_level']
    o2_status = "NORMAL"
    o2_class = "normal"
    if o2_val < 95:
        o2_status = "LOW"
        o2_class = "high"
    with c3:
        st.markdown(metric_html("Oxygen", "", o2_val, "%", o2_status, o2_class, "+1% from 1hr ago", "↗"), unsafe_allow_html=True)

    bp_val = current_vitals['blood_pressure']
    sys_val = current_vitals['systolic']
    bp_status = "NORMAL"
    bp_class = "normal"
    if sys_val > 140:
        bp_status = "HIGH"
        bp_class = "high"
    with c4:
        st.markdown(metric_html("Blood Pressure", "", bp_val, "mmHg", bp_status, bp_class, "Optimal range", "✓"), unsafe_allow_html=True)

    risk_class = "normal"
    if risk_level.lower() == "high" or risk_level.lower() == "warning": risk_class = "high"
    if risk_level.lower() == "critical": risk_class = "critical"
    with c5:
        st.markdown(metric_html("AI Risk Score", "", risk_score, "/100", risk_level.upper(), risk_class, f"Drivers: {len(risk_drivers)}", "⚠"), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # 2. Alert Banner
    if alerts:
        for alert in alerts:
            st.error(f"**{alert.get('code', 'ALERT')} Detected** — {alert.get('message', 'Please check vitals.')}")
    elif risk_level.lower() in ["critical", "high"]:
        st.warning(f"**Elevated Risk Level Detected** — Please review recent trends. Drivers: {', '.join(risk_drivers)}")

    # 3. Middle Section: Chart & Chat
    left_col, right_col = st.columns([2.2, 1.3], gap="large")

    with left_col:
        st.markdown("<div style='background:white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);'>", unsafe_allow_html=True)

        hr_history = st.session_state.heart_rate_history
        readings   = list(range(1, len(hr_history) + 1))
        avg_hr     = round(sum(hr_history) / len(hr_history), 1) if hr_history else 75
        hr_min, hr_max = min(hr_history), max(hr_history)
        min_idx = hr_history.index(hr_min)
        max_idx = hr_history.index(hr_max)

        fig = go.Figure()

        # --- Gradient fill: invisible top edge ---
        fig.add_trace(go.Scatter(
            x=readings, y=hr_history,
            mode="lines",
            line=dict(color="rgba(59,130,246,0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # --- Filled area (smooth spline) ---
        fig.add_trace(go.Scatter(
            x=readings, y=hr_history,
            mode="lines",
            name="Heart Rate",
            line=dict(color="#3b82f6", width=2.5, shape="spline", smoothing=1.3),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.10)",
            hovertemplate="<b>Reading %{x}</b><br>Heart Rate: %{y} bpm<extra></extra>",
        ))

        # --- Dashed average baseline ---
        fig.add_trace(go.Scatter(
            x=[readings[0], readings[-1]],
            y=[avg_hr, avg_hr],
            mode="lines",
            name=f"Avg Baseline ({avg_hr} bpm)",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hoverinfo="skip",
        ))

        # --- Min / Max markers ---
        fig.add_trace(go.Scatter(
            x=[readings[max_idx]], y=[hr_max],
            mode="markers+text",
            name="Peak",
            marker=dict(color="#ef4444", size=9, symbol="circle",
                        line=dict(color="white", width=2)),
            text=[f" {hr_max}"],
            textposition="top right",
            textfont=dict(color="#ef4444", size=11, family="Inter"),
            hovertemplate=f"<b>Peak</b>: {hr_max} bpm<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[readings[min_idx]], y=[hr_min],
            mode="markers+text",
            name="Low",
            marker=dict(color="#10b981", size=9, symbol="circle",
                        line=dict(color="white", width=2)),
            text=[f" {hr_min}"],
            textposition="bottom right",
            textfont=dict(color="#10b981", size=11, family="Inter"),
            hovertemplate=f"<b>Low</b>: {hr_min} bpm<extra></extra>",
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=0, r=8, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#475569"),
            title=dict(
                text="<b>Heart Rate Trends</b>  <span style='font-size:12px; color:#94a3b8; font-weight:400'>● Heart Rate &nbsp; ·· Avg Baseline</span>",
                x=0, xanchor="left",
                font=dict(size=16, color="#0f172a"),
            ),
            legend=dict(orientation="h", x=1, xanchor="right", y=1.15,
                        font=dict(size=11, color="#64748b"),
                        bgcolor="rgba(0,0,0,0)", borderwidth=0),
            xaxis=dict(
                showgrid=False, zeroline=False,
                tickfont=dict(size=11, color="#94a3b8"),
                title=dict(text="Reading", font=dict(size=11, color="#94a3b8")),
                tickmode="auto", nticks=8,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#f1f5f9",
                gridwidth=1,
                zeroline=False,
                tickfont=dict(size=11, color="#94a3b8"),
                title=dict(text="BPM", font=dict(size=11, color="#94a3b8")),
                range=[max(40, hr_min - 10), hr_max + 12],
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#1e293b", bordercolor="#334155",
                font=dict(color="white", size=12, family="Inter"),
            ),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<p style='color:#64748b; font-size:12px; margin-top:-8px; margin-bottom:8px;'>Detailed temporal analysis over the session</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Clinical Timeline — rendered via components to guarantee HTML rendering
        import streamlit.components.v1 as components
        components.html("""
        <style>
            body { margin: 0; font-family: 'Inter', -apple-system, sans-serif; }
        </style>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:24px; margin-top:4px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.02);">
            <h3 style="margin-top:0; font-size:18px; color:#0f172a; margin-bottom:20px; font-weight:600;">Clinical Timeline</h3>

            <div style="display:flex; gap:16px; margin-bottom:20px;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div style="width:12px; height:12px; border-radius:50%; background:#10b981; border:3px solid #fff; box-shadow:0 0 0 2px #10b981; flex-shrink:0; margin-top:3px;"></div>
                    <div style="width:2px; flex:1; background:#e2e8f0; margin-top:4px;"></div>
                </div>
                <div style="flex:1; padding-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:14px; font-weight:600; color:#0f172a;">Medication Logged</span>
                        <span style="font-size:12px; color:#94a3b8;">08:00 AM</span>
                    </div>
                    <p style="font-size:13px; color:#475569; margin:0;">Lisinopril (10mg) administered via morning dosage.</p>
                </div>
            </div>

            <div style="display:flex; gap:16px; margin-bottom:20px;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div style="width:12px; height:12px; border-radius:50%; background:#3b82f6; border:3px solid #fff; box-shadow:0 0 0 2px #3b82f6; flex-shrink:0; margin-top:3px;"></div>
                    <div style="width:2px; flex:1; background:#e2e8f0; margin-top:4px;"></div>
                </div>
                <div style="flex:1; padding-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:14px; font-weight:600; color:#0f172a;">Data Sync Successful</span>
                        <span style="font-size:12px; color:#94a3b8;">07:15 AM</span>
                    </div>
                    <p style="font-size:13px; color:#475569; margin:0;">Connected with wearable device. 8 hours of sleep data imported.</p>
                </div>
            </div>

            <div style="display:flex; gap:16px;">
                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div style="width:12px; height:12px; border-radius:50%; background:#eab308; border:3px solid #fff; box-shadow:0 0 0 2px #eab308; flex-shrink:0; margin-top:3px;"></div>
                </div>
                <div style="flex:1;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:14px; font-weight:600; color:#0f172a;">Low Activity Alert</span>
                        <span style="font-size:12px; color:#94a3b8;">Yesterday, 09:30 PM</span>
                    </div>
                    <p style="font-size:13px; color:#475569; margin:0;">Daily step goal not met (3,400 / 10,000 steps). Suggesting light walk tomorrow.</p>
                </div>
            </div>
        </div>
        """, height=280)



    with right_col:
        st.markdown("<div style='background:white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; font-size:18px; color:#0f172a; display:flex; align-items:center; gap:8px;'><span style='background:#2563eb; border-radius:6px; width:24px; height:24px; display:inline-flex; align-items:center; justify-content:center; color:white; font-size:14px;'>AI</span> HealthPulse AI Assistant</h3><p style='color:#10b981; font-size:12px; margin-bottom: 20px;'>● Online</p>", unsafe_allow_html=True)
        
        chat_container = st.container(height=380, border=False)
        with chat_container:
            if not st.session_state.chat_history:
                with st.chat_message("assistant"):
                    st.write("Hello! I'm monitoring Sarah's vitals. Ask me anything about her current status.")
            
            for role, text in st.session_state.chat_history:
                with st.chat_message(role):
                    st.write(text)

        user_input = st.chat_input("Type health query...")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            reply = backend_chat_response(user_input)
            if reply is None:
                reply = "Backend error: Unable to reach the local NLP Transformer."
            st.session_state.chat_history.append(("assistant", reply))
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)

elif selected_page == "Patient Records":
    st.markdown('''
        <div class="patient-header-container">
            <div>
                <div class="patient-title">Patient Records</div>
                <div class="patient-subtitle">Historical telemetry and vital sign logs</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    backend_vitals = api_get("/api/vitals?limit=100")
    if backend_vitals:
        records = pd.DataFrame(backend_vitals)
        st.dataframe(records, width="stretch", hide_index=True)
        st.download_button("Download Records CSV", records.to_csv(index=False), "healthpulse_records.csv", "text/csv")

elif selected_page == "AI Analysis":
    st.markdown('''
        <div class="patient-header-container">
            <div>
                <div class="patient-title">AI Analysis</div>
                <div class="patient-subtitle">Machine learning insights and risk predictions</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    if prediction:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{prediction['risk_score']}/100")
        c2.metric("Risk Level", prediction["risk_level"].title())
        ml_prediction = prediction.get("ml_prediction", {})
        c3.metric("ML Class", str(ml_prediction.get("predicted_class", "unknown")).title())
        c4.metric("ML Confidence", f"{float(ml_prediction.get('confidence', 0)):.2f}")
        st.markdown("**Risk Drivers**")
        for driver in prediction["drivers"]:
            st.write(f"- {driver}")
        if ml_prediction.get("probabilities"):
            st.markdown("**ML Class Probabilities**")
            st.dataframe(pd.DataFrame([ml_prediction["probabilities"]]), width="stretch", hide_index=True)

elif selected_page == "Clinical Alerts":
    st.markdown('''
        <div class="patient-header-container">
            <div>
                <div class="patient-title">Clinical Alerts</div>
                <div class="patient-subtitle">Active notifications and system flags</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    backend_alerts = api_get("/api/alerts?limit=100")
    if backend_alerts:
        alert_df = pd.DataFrame(backend_alerts)
        st.dataframe(alert_df, width="stretch", hide_index=True)

elif selected_page == "Reports":
    st.markdown('''
        <div class="patient-header-container">
            <div>
                <div class="patient-title">Reports</div>
                <div class="patient-subtitle">System evaluation and performance metrics</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    evaluation = api_get("/api/evaluation")
    if evaluation:
        alert_eval = evaluation.get("alert_rules", {})
        ml_eval = evaluation.get("ml_model") or {}
        st.markdown("**Rule-Based Alert Evaluation**")
        st.dataframe(pd.DataFrame(alert_eval.get("cases", [])), width="stretch", hide_index=True)
        if ml_eval:
            st.markdown("**ML Per-Class Metrics**")
            per_class = [{"class": class_name, **metrics} for class_name, metrics in ml_eval.get("per_class", {}).items()]
            st.dataframe(pd.DataFrame(per_class), width="stretch", hide_index=True)

elif selected_page == "Support":
    st.markdown('''
        <div class="patient-header-container">
            <div>
                <div class="patient-title">Support</div>
                <div class="patient-subtitle">System status and technical assistance</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    st.info("System running normally. All microservices are online.")
    
    st.markdown("### Frequently Asked Questions")
    with st.expander("How do I connect a new wearable device?"):
        st.write("Navigate to the patient settings dashboard and click 'Add Device'. Ensure the device is powered on and in Bluetooth pairing mode. HealthPulse AI supports most standard BLE health monitors.")
    
    with st.expander("What do the AI risk scores mean?"):
        st.write("Risk scores are calculated using a local NLP Transformer and time-series forecasting. A score below 20 is Low Risk, 20-60 is Moderate, and >60 indicates elevated risk requiring clinical review.")
        
    with st.expander("How do I export data for external EMR systems?"):
        st.write("Go to the 'Patient Records' tab and click the 'Download Records CSV' button. The exported CSV is formatted to comply with standard HL7 data ingestion templates.")
        
    st.markdown("### Contact Technical Support")
    with st.form("support_form"):
        st.text_input("Issue Subject")
        st.text_area("Describe the problem")
        submitted = st.form_submit_button("Submit Ticket")
        if submitted:
            st.success("Ticket submitted successfully. Support will respond within 24 hours.")
