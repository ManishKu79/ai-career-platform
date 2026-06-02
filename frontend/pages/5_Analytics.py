

import streamlit as st
import pandas as pd
from components.api_client import client
from components.charts import (
    top_skills_bar,
    score_histogram,
    funnel_chart,
    tier_donut_chart,
)

st.set_page_config(
    page_title="Analytics", page_icon="📊", layout="wide"
)
st.title("📊 Analytics Dashboard")

job_id = st.session_state.get("selected_job_id")

# ── Platform Health ────────────────────────────────────────────────────
st.subheader("Platform Health")

health_result = client.get_health()
db_stats      = client.get_db_stats()

if health_result["success"]:
    health = health_result["data"]
    status = health.get("status", "unknown")

    if status == "healthy":
        st.success("✅ All systems operational")
    else:
        st.warning(f"⚠️ System status: {status}")

    checks = health.get("checks", {})
    col1, col2, col3, col4 = st.columns(4)

    ping = checks.get("ping", {})
    col1.metric(
        "DB Ping",
        "✅ OK" if ping.get("status") == "ok" else "❌ Failed",
        delta=f"{ping.get('latency_ms', 0):.1f}ms" if ping.get("latency_ms") else None,
        delta_color="off",
    )

    write_ok = checks.get("write", {}).get("status") == "ok"
    col2.metric("Write Test", "✅ OK" if write_ok else "❌ Failed")

    read_ok = checks.get("read",  {}).get("status") == "ok"
    col3.metric("Read Test",  "✅ OK" if read_ok else "❌ Failed")

    idx = checks.get("indexes", {})
    missing_idx = idx.get("missing_indexes", [])
    col4.metric(
        "Indexes",
        "✅ All Present" if not missing_idx else f"⚠️ {len(missing_idx)} Missing",
    )

st.divider()

# ── Collection Stats ───────────────────────────────────────────────────
if db_stats["success"]:
    stats = db_stats["data"]
    db_info   = stats.get("database", {})
    coll_info = stats.get("collections", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Resumes",  coll_info.get("resumes",  {}).get("total", 0))
    col2.metric("Active Jobs",    coll_info.get("jobs",     {}).get("active", 0))
    col3.metric("Score Records",  coll_info.get("scores",   {}).get("total", 0))
    col4.metric("Data Size",      f"{db_info.get('data_size_mb', 0)} MB")
    col5.metric("Total Indexes",  db_info.get("index_count", 0))

st.divider()

# ── Pipeline Funnel ────────────────────────────────────────────────────
st.subheader("Processing Pipeline")

pipeline_result = client.get_pipeline_summary()

if pipeline_result["success"]:
    funnel_data = pipeline_result["data"].get("pipeline_funnel", {})
    stages = funnel_data.get("funnel_stages", [])

    if stages:
        col_funnel, col_table = st.columns([1, 1])

        with col_funnel:
            st.plotly_chart(funnel_chart(stages), use_container_width=True)

        with col_table:
            st.subheader("Stage Counts")
            stage_df = pd.DataFrame(stages)[["stage", "count", "percentage"]]
            stage_df.columns = ["Stage", "Count", "% of Total"]
            st.dataframe(stage_df, use_container_width=True, hide_index=True)

st.divider()

# ── Top Skills ─────────────────────────────────────────────────────────
st.subheader("Top Skills Across All Resumes")

skills_result = client.get_top_skills(limit=20)

if skills_result["success"]:
    skills_data = skills_result["data"]
    if skills_data:
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            st.plotly_chart(top_skills_bar(skills_data), use_container_width=True)

        with col_table:
            skills_df = pd.DataFrame(skills_data)[["skill", "count", "percentage"]]
            skills_df.columns = ["Skill", "Count", "% of Resumes"]
            st.dataframe(skills_df, use_container_width=True, hide_index=True)

st.divider()

# ── Job-Specific Analytics (requires selected job) ─────────────────────
if job_id:
    job_title = st.session_state.get("selected_job_title", "Selected Job")
    st.subheader(f"Analytics for: {job_title}")

    col_dist, col_gap = st.columns(2)

    # Score distribution
    with col_dist:
        dist_result = client.get_score_distribution(job_id)
        if dist_result["success"]:
            dist = dist_result["data"]
            histogram_data = dist.get("histogram_data", [])
            mean_score     = dist.get("mean_score", 0)

            if histogram_data:
                st.plotly_chart(
                    score_histogram(histogram_data, mean_score),
                    use_container_width=True
                )

                tier_dist = dist.get("tier_distribution", {})
                if tier_dist:
                    st.plotly_chart(
                        tier_donut_chart(tier_dist),
                        use_container_width=True
                    )

    # Skill gap heatmap
    with col_gap:
        gap_result = client.get_skill_gap_heatmap(job_id)
        if gap_result["success"]:
            gap_data = gap_result["data"]

            coverage = gap_data.get("skill_coverage", [])
            gaps     = gap_data.get("skill_gaps", [])

            if coverage:
                st.markdown("**Skill Coverage (% of candidates with skill)**")
                cov_df = pd.DataFrame(coverage)[["skill", "coverage_pct"]]
                cov_df.columns = ["Skill", "Coverage %"]
                cov_df = cov_df.sort_values("Coverage %", ascending=False)
                st.dataframe(cov_df, use_container_width=True,
                             hide_index=True, height=200)

            if gaps:
                st.markdown("**Top Skill Gaps (% of candidates missing)**")
                gap_df = pd.DataFrame(gaps)[["skill", "gap_pct"]]
                gap_df.columns = ["Skill", "Gap %"]
                gap_df = gap_df.sort_values("Gap %", ascending=False)
                st.dataframe(gap_df, use_container_width=True,
                             hide_index=True, height=200)

    # Hiring funnel
    funnel_result = client.get_hiring_funnel(job_id)
    if funnel_result["success"]:
        funnel = funnel_result["data"]
        st.subheader("Hiring Funnel")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Scored",     funnel.get("total_scored", 0))
        col2.metric("Passed Threshold", funnel.get("passed_threshold", 0))
        col3.metric("Priority Tier",    funnel.get("priority_tier", 0))
        col4.metric("Avg Score",
                    f"{funnel.get('avg_score', 0):.1f}%")

        conversion = funnel.get("conversion_rates", {})
        if conversion:
            st.markdown(
                f"**Conversion Rates:** "
                f"Scored → Threshold: `{conversion.get('scored_to_threshold', 0):.1f}%` | "
                f"Threshold → Priority: `{conversion.get('threshold_to_priority', 0):.1f}%`"
            )
else:
    st.info("👈 Select a job from the sidebar to see job-specific analytics.")
