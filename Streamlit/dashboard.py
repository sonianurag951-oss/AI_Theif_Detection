import io
import json
import math
import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from Sql.fetch_data import fetch_customer_reading, get_sample_customers
except Exception:
    fetch_customer_reading = None
    get_sample_customers = None


def load_customer_reading(customer_id=None, meter_id=None):
    """Fetch 26-day readings via FastAPI /readings if available, or direct data fallback."""
    if check_api("/health"):
        params = {}
        if customer_id:
            params["customer_id"] = customer_id.strip()
        if meter_id:
            params["meter_id"] = meter_id.strip()
        try:
            r = requests.get(f"{api_url}/readings", params=params, timeout=6)
            if r.ok:
                return r.json()
        except Exception:
            pass

    if fetch_customer_reading:
        return fetch_customer_reading(customer_id=customer_id, meter_id=meter_id)

    return {"found": False, "error": "Customer search service is unavailable."}


st.set_page_config(
    page_title="GridGuard | Electricity Theft Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .risk-badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    .gg-light {
        color: #111827;
        color-scheme: light;
    }
</style>
""", unsafe_allow_html=True)

DAY_COLS = [f"day_{i}" for i in range(1, 27)]


def kpi_card(label, value, unit="", sub="", value_color="#111827", sub_color="#111827"):
    st.markdown(f"""
<div class="gg-light" style="background-color:#fffbeb; border:1.5px solid #fde047; border-radius:8px; padding:10px 4px; text-align:center;">
<div style="font-size:11px; font-weight:800; text-transform:uppercase; color:#854d0e !important;">{label}</div>
<div style="font-size:20px; font-weight:900; margin-top:3px; color:{value_color} !important;">{value} <span style="font-size:11px; font-weight:700;">{unit}</span></div>
<div style="font-size:10.5px; font-weight:700; color:{sub_color} !important;">{sub}</div>
</div>
""", unsafe_allow_html=True)


def record_badge(flag):
    """Status badge with inline styles so the colors always apply."""
    base = (
        "display:inline-block; padding:5px 12px; border-radius:8px; "
        "font-weight:900; font-size:13px; line-height:1.5; border:1.5px solid; "
        "color:#ffffff;"
    )
    if flag == 1:
        return f'<span style="{base} background-color:#b91c1c; border-color:#7f1d1d;">🚨 Confirmed Theft Record (FLAG = 1)</span>'
    if flag == 0:
        return f'<span style="{base} background-color:#15803d; border-color:#14532d;">🟢 Normal Profile Record (FLAG = 0)</span>'
    return f'<span style="{base} background-color:#475569; border-color:#334155;">⚪ Unlabelled Customer</span>'


def dossier_card(meta, hist_badge):
    source = meta.get("source", "PostgreSQL Database")
    raw = f"""
        <div class="gg-light" style="background-color:#fef9c3; border:2.5px solid #eab308; border-radius:12px; padding:18px 22px; margin:14px 0 16px 0; color:#000000; box-shadow:0 4px 8px -1px rgba(0,0,0,0.1); box-sizing:border-box; width:100%; overflow:hidden;">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; border-bottom:2px solid #fde047; padding-bottom:10px; margin-bottom:14px;">
                <div style="font-size:18px; font-weight:900; color:#000000;">⚡ Utility Smart Meter Account Dossier</div>
                <div style="font-size:13px; font-weight:800; background-color:#ffffff; color:#000000; padding:4px 14px; border-radius:20px; border:1.5px solid #ca8a04; white-space:nowrap;">🗄️ Source: <span style="font-weight:900; color:#000000;">{source}</span></div>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:12px 20px; font-size:14.5px; line-height:1.6; color:#000000;">
                <div style="overflow-wrap:anywhere;"><span style="font-weight:800; color:#000000;">Customer ID:</span> <code style="background-color:#ffffff; color:#000000; font-weight:900; font-size:14px; padding:2px 8px; border-radius:5px; border:1.5px solid #ca8a04; overflow-wrap:anywhere;">{meta.get('customer_id')}</code></div>
                <div style="overflow-wrap:anywhere;"><span style="font-weight:800; color:#000000;">Meter Serial ID:</span> <code style="background-color:#ffffff; color:#000000; font-weight:900; font-size:14px; padding:2px 8px; border-radius:5px; border:1.5px solid #ca8a04; overflow-wrap:anywhere;">{meta.get('meter_id')}</code></div>
                <div><span style="font-weight:800; color:#000000;">Tariff Classification:</span> <span style="font-weight:900; color:#000000;">{meta.get('tariff_category', 'Residential')}</span></div>
                <div><span style="font-weight:800; color:#000000;">Grid Node / Region:</span> <span style="font-weight:900; color:#000000;">{meta.get('region', 'Madhya Pradesh')}</span></div>
                <div><span style="font-weight:800; color:#000000;">Hardware Meter Type:</span> <span style="font-weight:900; color:#000000;">{meta.get('meter_type', 'Smart')}</span></div>
                <div><span style="font-weight:800; color:#000000;">Commission Date:</span> <span style="font-weight:900; color:#000000;">{meta.get('installation_date', 'N/A')}</span></div>
                <div style="grid-column:1 / -1; margin-top:6px; padding-top:10px; border-top:1.5px dashed #ca8a04; color:#000000; display:flex; align-items:center; flex-wrap:wrap; gap:10px;"><span style="font-weight:800; color:#000000;">Historical Database Record:</span> {hist_badge}</div>
            </div>
        </div>
    """
    # Remove indentation and blank lines so Markdown never treats the HTML as a code block
    html = "\n".join(line.strip() for line in raw.splitlines() if line.strip())
    st.markdown(html, unsafe_allow_html=True)


st.sidebar.title("⚡ GridGuard")
st.sidebar.caption("Advanced AI Electricity Theft Detection")

api_url = st.sidebar.text_input("FastAPI Address", "http://127.0.0.1:8000").rstrip("/")
ollama_model = st.sidebar.text_input("Ollama LLM Model", "llama3.2:3b")
risk_threshold = st.sidebar.slider("Inspection Threshold", 0.05, 0.60, 0.20, 0.01)
use_ollama = st.sidebar.toggle("Enable AI Inspection Notes", value=True)

st.sidebar.divider()


# API Health Check
def check_api(path, timeout=4):
    try:
        r = requests.get(f"{api_url}{path}", timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


health = check_api("/health")

if health:
    st.sidebar.success(f"● API Online ({health.get('features', 0)} Features)")
    ollama_online = health.get("ollama", False)
    st.sidebar.caption(f"Ollama Engine: {'🟢 **Connected**' if ollama_online else '🔴 **Offline**'}")
    if not health.get("explanations_ready", False):
        st.sidebar.warning("⚠️ Baseline typical_values.json missing. Run `python -m src.ollama` once.")
else:
    st.sidebar.error("🔴 API Offline")
    st.sidebar.caption("Run: `uvicorn api:app --port 8000`")

st.sidebar.divider()


def api_post(path, json_data=None, params=None, timeout=60):
    try:
        r = requests.post(f"{api_url}{path}", json=json_data, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def ask_copilot(prompt_text, customer_context):
    system_msg = (
        "You are GridGuard's electricity theft investigator assistant. "
        "Analyze the customer's 26-day smart meter readings, SHAP risk drivers, "
        "and recommend technical audit steps (e.g., CT ratio check, tamper seal check, neutral bypass inspection). "
        "Be professional, technical, and concise."
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Context:\n{customer_context}\n\nQuestion: {prompt_text}"}
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 300}
            },
            timeout=120
        )
        return resp.json()["message"]["content"].strip()
    except Exception:
        return "⚠️ Ollama is not responding. Ensure Ollama is running (`ollama run llama3.2:3b`)."


def get_preset_readings(scenario: str):
    rng = np.random.default_rng(42)
    if scenario == "Normal Consumer":
        # Stable consumption between 16 and 26 kWh
        return np.round(rng.normal(21.0, 3.2, 26).clip(12.0, 32.0), 2).tolist()
    # elif scenario == "Sudden Meter Drop (Tampering)":
    #     # 13 days of normal consumption, then an immediate 80% collapse
    #     first_half = rng.normal(24.0, 2.5, 13).clip(18.0, 30.0)
    #     second_half = rng.normal(3.5, 1.2, 13).clip(0.5, 6.0)
    #     return np.round(np.concatenate([first_half, second_half]), 2).tolist()
    elif scenario == "Meter Bypass (Zero Streak)":
        # Normal for 8 days, then 18 continuous zeros
        active = rng.normal(20.0, 3.0, 8).clip(14.0, 28.0)
        zeros = np.zeros(18)
        return np.round(np.concatenate([active, zeros]), 2).tolist()
    # elif scenario == "Intermittent Under-recording":
    #     # High variability, sudden drop-offs and low consumption
    #     base = rng.normal(18.0, 4.0, 26)
    #     drop_indices = [3, 4, 10, 11, 12, 18, 19, 20, 21]
    #     base[drop_indices] = rng.uniform(0.0, 2.0, len(drop_indices))
    #     return np.round(np.clip(base, 0.0, 30.0), 2).tolist()
    else:
        return [15.0] * 26


# SESSION STATE INITIALIZATION
if "cust_id_input" not in st.session_state:
    st.session_state["cust_id_input"] = "7ABCEC7B9CB9E49B2AC3BA29B21585EB"
if "meter_id_input" not in st.session_state:
    st.session_state["meter_id_input"] = "MTR01077"
if "customer_meta" not in st.session_state:
    st.session_state["customer_meta"] = None
if "customer_readings" not in st.session_state:
    # Attempt to load default customer readings, fallback to default 26 readings
    init_res = load_customer_reading(
        customer_id=st.session_state["cust_id_input"],
        meter_id=st.session_state["meter_id_input"]
    )
    if init_res and init_res.get("found"):
        st.session_state["customer_readings"] = init_res["readings"]
        st.session_state["customer_meta"] = init_res
    else:
        st.session_state["customer_readings"] = [15.0] * 26
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "copilot_chat" not in st.session_state:
    st.session_state["copilot_chat"] = []

# HEADER
st.title("⚡ Electricity Theft Diagnostic Intelligence")
st.markdown("Analyze customer 26-day smart meter consumption patterns, calculate theft probability, and generate AI field inspection memos.")

# MAIN TABS
tab_diagnose, tab_copilot, tab_model_metrics = st.tabs([
    "🔍 Single Customer Diagnostic",
    "🤖 AI Investigation Copilot",
    "📈 Model Benchmark & CV"
])


# SINGLE CUSTOMER
with tab_diagnose:
    st.subheader("1. Customer & Meter Lookup")
    st.caption("Enter a **Customer ID** or **Meter ID** to retrieve official 26-day smart meter consumption readings from the database.")

    col_c1, col_c2, col_btn = st.columns([2.2, 2.0, 1.2])
    cust_id = col_c1.text_input(
        "Customer ID",
        value=st.session_state.get("cust_id_input", ""),
        help="Unique customer identifier (e.g. A0E791400CF1C48C43DC26A68227854A)"
    )
    meter_id = col_c2.text_input(
        "Meter Serial Number (Meter ID)",
        value=st.session_state.get("meter_id_input", ""),
        help="Hardware Smart Meter ID (e.g. MTR00001)"
    )

    with col_btn:
        st.write("")
        st.write("")
        fetch_btn = st.button("📥 Fetch Readings", type="primary", use_container_width=True)

    # Database Sample Selector
    sample_records = get_sample_customers(limit=10) if get_sample_customers else []
    if sample_records:
        sample_labels = ["-- Select a Verified Utility Customer to Test --"] + [
            f"{s['customer_id']} | {s['meter_id']} ({'🚨 Historical Theft' if s['flag'] == 1 else '🟢 Normal Record'}) [{s['tariff_category']}]"
            for s in sample_records
        ]
        selected_sample = st.selectbox(
            "Or quick-select a customer from utility records:",
            sample_labels,
            index=0
        )
        if selected_sample != "-- Select a Verified Utility Customer to Test --":
            parts = selected_sample.split(" | ")
            s_cid = parts[0].strip()
            s_mid = parts[1].split(" ")[0].strip()
            if s_cid != st.session_state.get("cust_id_input") or s_mid != st.session_state.get("meter_id_input"):
                fetched = load_customer_reading(customer_id=s_cid, meter_id=s_mid)
                if fetched and fetched.get("found"):
                    st.session_state["cust_id_input"] = fetched["customer_id"]
                    st.session_state["meter_id_input"] = fetched["meter_id"]
                    st.session_state["customer_readings"] = fetched["readings"]
                    st.session_state["customer_meta"] = fetched
                    st.session_state["analysis_result"] = None
                    st.rerun()

    if fetch_btn:
        if not cust_id and not meter_id:
            st.warning("⚠️ Please provide at least a Customer ID or Meter ID to fetch readings.")
        else:
            with st.spinner("Fetching smart meter readings from database/dataset..."):
                fetched = load_customer_reading(customer_id=cust_id, meter_id=meter_id)
                if fetched and fetched.get("found"):
                    st.session_state["cust_id_input"] = fetched["customer_id"]
                    st.session_state["meter_id_input"] = fetched["meter_id"]
                    st.session_state["customer_readings"] = fetched["readings"]
                    st.session_state["customer_meta"] = fetched
                    st.session_state["analysis_result"] = None
                    st.success(f"✅ Loaded 26-day readings for Customer `{fetched['customer_id']}` (Meter `{fetched['meter_id']}`) from {fetched.get('source', 'records')}!")
                    st.rerun()
                else:
                    st.error(f"❌ {fetched.get('error', 'Customer not found.')}")

    # Active Customer Account Profile Banner
    meta = st.session_state.get("customer_meta")
    if not meta:
        meta = {
            "customer_id": cust_id or st.session_state.get("cust_id_input", "7ABCEC7B9CB9E49B2AC3BA29B21585EB"),
            "meter_id": meter_id or st.session_state.get("meter_id_input", "MTR01077"),
            "tariff_category": "Residential",
            "region": "Madhya Pradesh",
            "meter_type": "Smart",
            "installation_date": "Active Record",
            "flag": 0,
            "source": "PostgreSQL Database"
        }

    hist_badge = record_badge(meta.get("flag"))

    # 1. ACCOUNT DOSSIER CARD
    dossier_card(meta, hist_badge)

    # 2. CONSUMPTION KEY PERFORMANCE METRICS
    if "customer_readings" not in st.session_state or not st.session_state["customer_readings"]:
        st.session_state["customer_readings"] = [15.0] * 26
    current_readings = [float(x) for x in st.session_state.get("customer_readings", [15.0] * 26)]
    tot_kwh = sum(current_readings)
    avg_kwh = float(np.mean(current_readings))
    peak_kwh = max(current_readings)
    peak_day = current_readings.index(peak_kwh) + 1
    min_kwh = min(current_readings)
    min_day = current_readings.index(min_kwh) + 1
    zero_count = sum(1 for x in current_readings if x <= 0)

    h1_mean = float(np.mean(current_readings[:13]))
    h2_mean = float(np.mean(current_readings[13:]))
    half_delta_pct = ((h2_mean - h1_mean) / (h1_mean + 1e-6)) * 100

    if half_delta_pct <= -35:
        trend_badge_text = f"📉 Drop ({half_delta_pct:.1f}%)"
        trend_color = "#dc2626"
    elif half_delta_pct >= 35:
        trend_badge_text = f"📈 Jump ({half_delta_pct:+.1f}%)"
        trend_color = "#2563eb"
    else:
        trend_badge_text = f"➡️ Stable ({half_delta_pct:+.1f}%)"
        trend_color = "#16a34a"

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        kpi_card("⚡ Total 26d Usage", f"{tot_kwh:.2f}", "kWh", f"{(tot_kwh / 26) * 30:.1f} kWh/mo est.")

    with k2:
        kpi_card("📊 Daily Average", f"{avg_kwh:.2f}", "kWh", "Typical baseline")

    with k3:
        kpi_card("🏔️ Peak Usage", f"{peak_kwh:.2f}", "kWh", f"Recorded on Day {peak_day}", sub_color="#d97706")

    with k4:
        kpi_card("📉 Lowest Usage", f"{min_kwh:.2f}", "kWh", f"Recorded on Day {min_day}", sub_color="#2563eb")

    with k5:
        kpi_card("🚫 Zero Days", zero_count, "/ 26d", f"{(zero_count / 26) * 100:.1f}% of period",
                 value_color="#dc2626" if zero_count > 0 else "#16a34a")

    with k6:
        kpi_card("🔄 H1 vs H2 Shift", trend_badge_text, "", f"{h1_mean:.1f} → {h2_mean:.1f} kWh",
                 value_color=trend_color)

    # 3. INTEGRATED LOAD CURVE GRAPH & DROP AUDIT
    st.write("")
    col_g1, col_g2 = st.columns([2.8, 1.2])

    with col_g1:
        st.markdown("##### 📈 26-Day Smart Meter Load Curve & Anomaly Detection (kWh)")
        chart_df = pd.DataFrame({
            "Day_Num": list(range(1, 27)),
            "Day": [f"Day {i}" for i in range(1, 27)],
            "Consumption": current_readings,
            "Is_Zero": [x <= 0 for x in current_readings],
            "Is_Peak": [x == peak_kwh for x in current_readings],
            "Deviation": [round(x - avg_kwh, 2) for x in current_readings]
        })

        base_c = alt.Chart(chart_df).encode(
            x=alt.X("Day_Num:Q", title="Inspection Period (Days 1 to 26)", scale=alt.Scale(domain=[1, 26])),
            y=alt.Y("Consumption:Q", title="Consumption (kWh)")
        )

        area_c = base_c.mark_area(
            line={'color': '#2563eb', 'width': 2.5},
            color=alt.Gradient(
                gradient='linear',
                stops=[
                    alt.GradientStop(color='#93c5fd', offset=0),
                    alt.GradientStop(color='rgba(255, 255, 255, 0.05)', offset=1)
                ],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            tooltip=[
                alt.Tooltip("Day_Num:Q", title="Day"),
                alt.Tooltip("Consumption:Q", title="Usage (kWh)", format=".2f"),
                alt.Tooltip("Deviation:Q", title="vs Average (kWh)", format="+.2f")
            ]
        )

        pts_c = base_c.mark_circle(size=55, color='#2563eb')

        avg_line_c = alt.Chart(pd.DataFrame({"avg": [avg_kwh]})).mark_rule(
            color='#d97706', strokeDash=[5, 5], strokeWidth=2
        ).encode(
            y='avg:Q',
            tooltip=[alt.Tooltip("avg:Q", title="Mean 26d Consumption (kWh)", format=".2f")]
        )

        zero_pts_c = alt.Chart(chart_df[chart_df["Is_Zero"]]).mark_circle(
            size=140, color="#dc2626"
        ).encode(
            x="Day_Num:Q", y="Consumption:Q",
            tooltip=[alt.Tooltip("Day_Num:Q", title="🚨 ZERO Reading Day!")]
        )

        peak_pt_c = alt.Chart(chart_df[chart_df["Is_Peak"]]).mark_circle(
            size=140, color="#f59e0b"
        ).encode(
            x="Day_Num:Q", y="Consumption:Q",
            tooltip=[alt.Tooltip("Day_Num:Q", title="🏔️ Peak Usage Day!"), alt.Tooltip("Consumption:Q", title="Peak (kWh)", format=".2f")]
        )

        st.altair_chart(area_c + pts_c + avg_line_c + zero_pts_c + peak_pt_c,use_container_width=True)

    with col_g2:
        st.markdown("##### ⚖️ Half-Period Drop Audit")
        h_df = pd.DataFrame({
            "Period": ["Days 1–13 (H1)", "Days 14–26 (H2)"],
            "Daily_Avg_kWh": [round(h1_mean, 2), round(h2_mean, 2)],
            "Color": ["#3b82f6", "#ef4444" if half_delta_pct <= -25 else "#10b981"]
        })
        h_bar = alt.Chart(h_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Period:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Daily_Avg_kWh:Q", title="Mean kWh / Day"),
            color=alt.Color("Color:N", scale=None),
            tooltip=["Period", "Daily_Avg_kWh"]
        ).properties(height=230)
        st.altair_chart(h_bar,use_container_width=True)
        st.caption(f"H1 Avg: **{h1_mean:.2f} kWh** | H2 Avg: **{h2_mean:.2f} kWh** ({half_delta_pct:+.1f}%)")


# AI INVESTIGATION COPILOT
with tab_copilot:
    st.subheader("🤖 AI Enterprise Theft Copilot")
    st.caption("Connected to Utility Smart Meter Database, Anomaly Telemetry, and ML Benchmarks.")

    # HISTORY MANAGEMENT CONTROLS
    hist_col1, hist_col2, hist_col3 = st.columns([2, 1, 1])

    with hist_col1:
        st.markdown(f"**Chat History**: `{len(st.session_state['copilot_chat'])}` interactions recorded")

    with hist_col2:
        # Download Chat History
        if st.session_state["copilot_chat"]:
            chat_text = "\n\n".join([
                f"### {'Investigator' if msg['role'] == 'user' else 'GridGuard AI'}:\n{msg['content']}"
                for msg in st.session_state["copilot_chat"]
            ])
            st.download_button(
                "📥 Export Transcript",
                data=chat_text,
                file_name="gridguard_investigation_log.md",
                mime="text/markdown",
                use_container_width=True
            )

    with hist_col3:
        # Clear Chat History Button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["copilot_chat"] = []
            st.rerun()

    st.divider()

    # QUICK QUESTION PROMPT BUTTONS
    st.markdown("##### 💡 Common Inquiries:")
    q_cols = st.columns(4)
    q1 = q_cols[0].button("📊 Top 10 High-Risk Customers", use_container_width=True)
    q2 = q_cols[1].button("🏆 Model Performance Comparison", use_container_width=True)
    q3 = q_cols[2].button("🔎 Why is the #1 Customer Flagged?", use_container_width=True)
    q4 = q_cols[3].button("⚡ Common Theft Patterns", use_container_width=True)

    # Get active customer context from Diagnostic tab
    current_res = st.session_state.get("analysis_result")
    active_cust_context = {
        "customer_id": current_res.get("customer_id") if current_res else "None",
        "theft_probability": f"{current_res.get('probability', 0):.1%}" if current_res else "None",
        "flagged": current_res.get("flagged") if current_res else False,
        "key_signals": current_res.get("reasons", []) if current_res else []
    }

    # Determine prompt to run
    user_query = None
    if q1:
        user_query = "List the top 10 customers with the highest theft risk from the database in a clean Markdown table with Rank, Customer ID, Theft Risk, Threat Level, Primary Anomaly Evidence, and Required Field Audit."
    elif q2:
        user_query = "Compare our trained models (Random Forest, XGBoost, Hybrid CNN) by PR-AUC, Recall, and Precision. Which one is best for production deployment?"
    elif q3:
        user_query = "Look at the #1 ranked customer in our database and explain what meter tampering signals caused their high risk score."
    elif q4:
        user_query = "What are the most frequent anomaly patterns (e.g., zero streaks, sudden drops) that trigger theft alerts across our grid?"

    # RENDER CHAT HISTORY MESSAGES
    for chat in st.session_state["copilot_chat"]:
        with st.chat_message(chat["role"], avatar="👤" if chat["role"] == "user" else "⚡"):
            st.markdown(chat["content"])

    # CHAT INPUT BOX
    chat_input = st.chat_input("Ask Anything about this project...")
    if chat_input:
        user_query = chat_input

    # PROCESS QUERY & GENERATE RESPONSE
    if user_query:
        # Add User Message to History
        st.session_state["copilot_chat"].append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Searching database & generating executive report..."):
                reply = ask_copilot(user_query, active_cust_context)
                st.markdown(reply)
                # Add Assistant Response to History
                st.session_state["copilot_chat"].append({"role": "assistant", "content": reply})


# MODEL BENCHMARK & PERFORMANCE
# with tab_model_metrics:
#     st.subheader("📈 Machine Learning Performance & Validation")
#     metrics = check_api("/metrics")

#     if metrics and metrics.get("test"):
#         test_df = pd.DataFrame(metrics["test"])
#         st.markdown("##### Held-Out Test Evaluation")
#         st.dataframe(test_df.round(4), hide_index=True, width="stretch")

#         if "pr_auc" in test_df.columns:
#             m_chart = alt.Chart(test_df).mark_bar(color="#2563eb").encode(
#                 x=alt.X("model:N", title="Candidate Architecture"),
#                 y=alt.Y("pr_auc:Q", title="PR-AUC Score", scale=alt.Scale(domain=[0, 1])),
#                 tooltip=["model", "pr_auc", "roc_auc"]
#             )
#             st.altair_chart(m_chart, width="stretch")

#         if metrics.get("cv"):
#             st.markdown("##### Stratified 5-Fold Cross-Validation")
#             cv_df = pd.DataFrame(metrics["cv"])
#             st.dataframe(cv_df.round(4), hide_index=True, width="stretch")
#     else:
#         st.info("Run `python src/training.py` once to train and log model comparison metrics.")
