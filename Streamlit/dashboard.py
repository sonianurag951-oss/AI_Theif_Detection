import io
import json
import math
import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

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
</style>
""", unsafe_allow_html=True)

DAY_COLS = [f"day_{i}" for i in range(1, 27)]

st.sidebar.title("⚡ GridGuard")
st.sidebar.caption("Advanced AI Electricity Theft Detection")

api_url = st.sidebar.text_input("FastAPI Address", "http://127.0.0.1:8000").rstrip("/")
ollama_model = st.sidebar.text_input("Ollama LLM Model", "llama3.2:3b")
risk_threshold = st.sidebar.slider("Inspection Threshold", 0.10, 0.90, 0.50, 0.05)
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
st.sidebar.caption("Decision-support system. Flagged scores indicate statistical anomalies, not legal proof.")


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
    elif scenario == "Sudden Meter Drop (Tampering)":
        # 13 days of normal consumption, then an immediate 80% collapse
        first_half = rng.normal(24.0, 2.5, 13).clip(18.0, 30.0)
        second_half = rng.normal(3.5, 1.2, 13).clip(0.5, 6.0)
        return np.round(np.concatenate([first_half, second_half]), 2).tolist()
    elif scenario == "Meter Bypass (Zero Streak)":
        # Normal for 8 days, then 18 continuous zeros
        active = rng.normal(20.0, 3.0, 8).clip(14.0, 28.0)
        zeros = np.zeros(18)
        return np.round(np.concatenate([active, zeros]), 2).tolist()
    elif scenario == "Intermittent Under-recording":
        # High variability, sudden drop-offs and low consumption
        base = rng.normal(18.0, 4.0, 26)
        drop_indices = [3, 4, 10, 11, 12, 18, 19, 20, 21]
        base[drop_indices] = rng.uniform(0.0, 2.0, len(drop_indices))
        return np.round(np.clip(base, 0.0, 30.0), 2).tolist()
    else:
        return [15.0] * 26


# SESSION STATE INITIALIZATION
if "customer_readings" not in st.session_state:
    st.session_state["customer_readings"] = get_preset_readings("Sudden Meter Drop (Tampering)")
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
    st.subheader("1. Customer & Meter Profile")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    cust_id = col_c1.text_input("Customer ID", value="CUST-49021", help="Unique utility account identifier")
    meter_id = col_c2.text_input("Meter Serial Number", value="MTR-8839-X", help="Physical smart meter hardware ID")
    tariff_type = col_c3.selectbox("Tariff Classification", ["Residential (Single-phase)", "Commercial (Three-phase)", "Industrial", "Agricultural"])

    st.markdown("#### Quick Test Presets")
    preset_cols = st.columns(4)
    if preset_cols[0].button("🟢 Normal Consumer", width="stretch"):
        st.session_state["customer_readings"] = get_preset_readings("Normal Consumer")
        st.rerun()
    if preset_cols[1].button("🔴 Sudden Drop (Tampering)", width="stretch"):
        st.session_state["customer_readings"] = get_preset_readings("Sudden Meter Drop (Tampering)")
        st.rerun()
    if preset_cols[2].button("🔴 Total Bypass (Zero Streak)", width="stretch"):
        st.session_state["customer_readings"] = get_preset_readings("Meter Bypass (Zero Streak)")
        st.rerun()
    if preset_cols[3].button("🟠 Intermittent Drops", width="stretch"):
        st.session_state["customer_readings"] = get_preset_readings("Intermittent Under-recording")
        st.rerun()


    # 26-DAY DAILY CONSUMPTION READINGS 
    col_sec_title, col_edit_btn = st.columns([3, 1])

    with col_sec_title:
        st.subheader("2. 26-Day Daily Consumption Readings (kWh)")
        st.caption("Click **✏️ Edit Day Reading** to modify any specific day, or edit the table directly:")

    with col_edit_btn:
        # EDIT BUTTON / POPUP
        with st.popover("✏️ Edit Day Reading", width="stretch"):
            st.markdown("##### ✏️ Modify Reading for a Day")
            day_options = [f"Day {i}" for i in range(1, 27)]
            selected_day = st.selectbox("Select Day to Edit:", day_options)
            
            # Get index (0 to 25)
            day_idx = int(selected_day.split()[1]) - 1
            current_val = float(st.session_state["customer_readings"][day_idx])
            
            new_val = st.number_input(
                f"Enter new kWh for {selected_day}:",
                min_value=0.0,
                max_value=200.0,
                value=current_val,
                step=0.5,
                format="%.2f"
            )
            
            if st.button("💾 Save Reading", type="primary", width="stretch"):
                st.session_state["customer_readings"][day_idx] = round(new_val, 2)
                st.success(f"✅ {selected_day} updated to {new_val:.2f} kWh!")
                st.rerun()

    # 1-row numeric DataFrame 
    readings_df = pd.DataFrame(
        [st.session_state["customer_readings"]],
        columns=[f"Day {i}" for i in range(1, 27)],
        index=["kWh"]
    )

    edited_df = st.data_editor(
        readings_df,
        width="stretch",
        num_rows="fixed",
        hide_index=False,
        key="consumption_grid_editor"
    )

    try:
        updated_readings = [float(x) for x in edited_df.iloc[0].tolist()]
        st.session_state["customer_readings"] = updated_readings
    except Exception:
        updated_readings = st.session_state["customer_readings"]

    st.write("")
    run_btn = st.button("🚀 Analyze Theft Risk & Generate AI Report", type="primary", width="stretch")

    if run_btn:
        with st.spinner("Executing ML pipeline, computing SHAP explainers & querying Llama 3.2..."):
            payload = {
                "customer_id": cust_id,
                "readings": updated_readings
            }
            res = api_post("/explain", json_data=payload, params={"use_ollama": str(use_ollama).lower()})
            if res:
                st.session_state["analysis_result"] = res

    # DASHBOARD
    result = st.session_state["analysis_result"]

    if result:
        st.divider()
        st.subheader("3. Investigation Intelligence Report")

        prob = result["probability"]
        flagged = result["flagged"]
        
        # Risk Badge Logic
        if prob >= 0.75:
            badge_html = f"<div class='risk-badge-critical'>🚨 CRITICAL THEFT RISK ({prob:.1%})</div>"
            recommendation = "Immediate physical site audit & meter replacement recommended."
        elif prob >= risk_threshold:
            badge_html = f"<div class='risk-badge-high'>⚠️ HIGH THEFT RISK ({prob:.1%})</div>"
            recommendation = "Assign field inspector for tamper seal and meter calibration check."
        elif prob >= 0.30:
            badge_html = f"<div class='risk-badge-high'>🟡 MODERATE RISK ({prob:.1%})</div>"
            recommendation = "Place meter on automated high-frequency telemetry monitoring."
        else:
            badge_html = f"<div class='risk-badge-low'>✅ NORMAL PROFILE ({prob:.1%})</div>"
            recommendation = "No inspection needed. Pattern consistent with standard consumption."

        # Top Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"**Risk Score**<br>{badge_html}", unsafe_allow_html=True)
        with m2:
            st.metric("Theft Probability", f"{prob:.1%}", delta=f"Threshold: {risk_threshold:.0%}")
        with m3:
            avg_kwh = float(np.mean(updated_readings))
            st.metric("Avg Daily Consumption", f"{avg_kwh:.2f} kWh")
        with m4:
            zero_count = int(sum(x <= 0 for x in updated_readings))
            st.metric("Zero Consumption Days", f"{zero_count} / 26 days")

        st.info(f"**Recommended Action:** {recommendation}")

        # Charts Section
        c_left, c_right = st.columns([3, 2])

        with c_left:
            st.markdown("##### 26-Day Consumption Trend (kWh)")
            chart_data = pd.DataFrame({
                "Day": list(range(1, 27)),
                "Consumption": updated_readings,
                "Is_Zero": [x <= 0 for x in updated_readings]
            })

            line = alt.Chart(chart_data).mark_line(color="#2563eb", strokeWidth=2.5, point=True).encode(
                x=alt.X("Day:Q", title="Inspection Period (Day)"),
                y=alt.Y("Consumption:Q", title="Consumption (kWh)"),
                tooltip=["Day", "Consumption"]
            )

            zeros = alt.Chart(chart_data[chart_data["Is_Zero"]]).mark_circle(
                color="#dc2626", size=110
            ).encode(x="Day:Q", y="Consumption:Q", tooltip=["Day", "Consumption"])

            mean_rule = alt.Chart(pd.DataFrame({"avg": [avg_kwh]})).mark_rule(
                color="#64748b", strokeDash=[4, 4]
            ).encode(y="avg:Q")

            st.altair_chart(line + zeros + mean_rule, width="stretch")

        with c_right:
            st.markdown("##### Strongest SHAP Risk Drivers")
            if result.get("top_features"):
                drivers_df = pd.DataFrame(result["top_features"])
                drivers_df["Direction"] = drivers_df["shap"].apply(lambda v: "Raises Risk" if v > 0 else "Lowers Risk")
                
                bar = alt.Chart(drivers_df).mark_bar().encode(
                    x=alt.X("shap:Q", title="Impact on Risk Score (SHAP)"),
                    y=alt.Y("label:N", sort="-x", title=None),
                    color=alt.Color("Direction:N", scale=alt.Scale(
                        domain=["Raises Risk", "Lowers Risk"],
                        range=["#ef4444", "#3b82f6"]
                    )),
                    tooltip=["label", "shap"]
                )
                st.altair_chart(bar, width="stretch")

        # AI Inspection Memo
        st.markdown("##### 📋 AI Field Inspection Memo (Generated by Llama 3.2)")
        note_source = result.get("note_source", "template").upper()
        
        st.success(result["note"])
        st.caption(f"Engine: **{note_source}** ({ollama_model if note_source == 'OLLAMA' else 'Statistical Template'})")

        # Detailed Risk Factors Table
        if result.get("reasons"):
            with st.expander("🔎 View Statistical Anomaly Details vs. Typical Utility Customer"):
                reasons_df = pd.DataFrame(result["reasons"])
                st.dataframe(
                    reasons_df.rename(columns={
                        "label": "Metric Description",
                        "value": "This Customer",
                        "typical": "Typical Baseline",
                        "shap": "SHAP Risk Contribution"
                    }),
                    hide_index=True,
                    width="stretch"
                )


#  AI INVESTIGATION COPILOT
with tab_copilot:
    st.subheader("🤖 AI Enterprise Theft Copilot")
    st.caption("Connected to Utility Smart Meter Database, Anomaly Telemetry, and ML Benchmarks.")

    #  HISTORY MANAGEMENT CONTROLS
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
                width="stretch"
            )

    with hist_col3:
        # Clear Chat History Button
        if st.button("🗑️ Clear History", width="stretch"):
            st.session_state["copilot_chat"] = []
            st.rerun()

    st.divider()

    # QUICK QUESTION PROMPT BUTTONS
    st.markdown("##### 💡 Common Inquiries:")
    q_cols = st.columns(4)
    q1 = q_cols[0].button("📊 Top 10 High-Risk Customers", width="stretch")
    q2 = q_cols[1].button("🏆 Model Performance Comparison", width="stretch")
    q3 = q_cols[2].button("🔎 Why is the #1 Customer Flagged?", width="stretch")
    q4 = q_cols[3].button("⚡ Common Theft Patterns", width="stretch")

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
with tab_model_metrics:
    st.subheader("📈 Machine Learning Performance & Validation")
    metrics = check_api("/metrics")

    if metrics and metrics.get("test"):
        test_df = pd.DataFrame(metrics["test"])
        st.markdown("##### Held-Out Test Evaluation")
        st.dataframe(test_df.round(4), hide_index=True, width="stretch")

        if "pr_auc" in test_df.columns:
            m_chart = alt.Chart(test_df).mark_bar(color="#2563eb").encode(
                x=alt.X("model:N", title="Candidate Architecture"),
                y=alt.Y("pr_auc:Q", title="PR-AUC Score", scale=alt.Scale(domain=[0, 1])),
                tooltip=["model", "pr_auc", "roc_auc"]
            )
            st.altair_chart(m_chart, width="stretch")

        if metrics.get("cv"):
            st.markdown("##### Stratified 5-Fold Cross-Validation")
            cv_df = pd.DataFrame(metrics["cv"])
            st.dataframe(cv_df.round(4), hide_index=True, width="stretch")
    else:
        st.info("Run `python src/training.py` once to train and log model comparison metrics.")