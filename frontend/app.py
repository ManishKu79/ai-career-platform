# frontend/app.py

import streamlit as st
from components.api_client import client, API_BASE_URL

# ── Page Configuration ─────────────────────────────────────────────────
# Must be the FIRST Streamlit command in the script
st.set_page_config(
    page_title="AI Career Intelligence Platform",
    page_icon="🎯",
    layout="wide",           # Use full browser width
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ──────────────────────────────────────
# Initialize all session state keys used across pages
# This prevents KeyError when pages access uninitialized state

defaults = {
    "selected_job_id":    None,   # Currently selected job
    "selected_job_title": None,   # Job title for display
    "uploaded_file_ids":  [],     # File IDs uploaded this session
    "last_pipeline_result": None, # Most recent pipeline result
    "api_connected":      False,  # API connectivity status
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Check API Connectivity ────────────────────────────────────────────
# Check once per session load (not on every re-run)
st.session_state.api_connected = client.check_api_health()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Career Platform")
    st.caption("AI-Powered ATS System")
    st.divider()

    # API Connection Status
    if st.session_state.api_connected:
        st.success("✅ API Connected")
    else:
        st.error(f"❌ API Offline\n\n`{API_BASE_URL}`")
        st.caption("Start the FastAPI server:\n```\nuvicorn backend.main:app --reload\n```")

    st.divider()

    # Job selector — persists across page navigation
    st.subheader("Active Job")

    if st.session_state.api_connected:
        jobs_result = client.list_jobs(active_only=True)

        if jobs_result["success"] and jobs_result["data"].get("jobs"):
            jobs      = jobs_result["data"]["jobs"]
            job_options = {j["title"]: j["job_id"] for j in jobs}

            selected_title = st.selectbox(
                "Select Job",
                options=list(job_options.keys()),
                index=None,
                placeholder="Choose a job...",
                key="job_selector",
            )

            if selected_title:
                st.session_state.selected_job_id    = job_options[selected_title]
                st.session_state.selected_job_title = selected_title
                st.caption(f"ID: `{st.session_state.selected_job_id[:8]}...`")
        else:
            st.info("No active jobs. Create one on the Jobs page.")

    st.divider()

    # Quick stats in sidebar
    if st.session_state.api_connected:
        stats_result = client.get_db_stats()
        if stats_result["success"]:
            coll = stats_result["data"].get("collections", {})
            resumes_stats = coll.get("resumes", {})
            jobs_stats    = coll.get("jobs", {})
            scores_stats  = coll.get("scores", {})

            st.subheader("Database")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Resumes",  resumes_stats.get("total", 0))
                st.metric("Scores",   scores_stats.get("total", 0))
            with col2:
                st.metric("Jobs",     jobs_stats.get("total", 0))

    st.divider()
    st.caption("v1.0.0 | MongoDB + FastAPI + Streamlit")

# ── Main Page ─────────────────────────────────────────────────────────
st.title("🎯 AI Career Intelligence Platform")
st.subheader("Production-Grade ATS with NLP-Driven Scoring")

st.markdown("""
### Welcome to the AI Career Intelligence Platform

This system automates resume screening using Natural Language Processing 
and machine learning to objectively rank candidates against job requirements.

#### Processing Pipeline
""")

# Pipeline flow diagram using columns
cols = st.columns(5)
stages = [
    ("📄", "Upload",   "PDF/DOCX parsing"),
    ("🔤", "NLP",      "Tokenization & NER"),
    ("🔧", "Skills",   "Taxonomy extraction"),
    ("📊", "Score",    "TF-IDF + Cosine"),
    ("🏆", "Rank",     "Multi-criteria ranking"),
]

for col, (icon, title, desc) in zip(cols, stages):
    with col:
        st.markdown(
            f"""
            <div style="
                background:#f0f2f6;
                border-radius:8px;
                padding:12px;
                text-align:center;
                height:100px;
            ">
                <div style="font-size:28px">{icon}</div>
                <div style="font-weight:bold">{title}</div>
                <div style="font-size:11px;color:#666">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.info("👈 Use the sidebar to select a job and navigate between pages.")