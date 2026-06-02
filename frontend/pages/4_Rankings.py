

import streamlit as st
import pandas as pd
from components.api_client import client
from components.charts import (
    score_gauge, component_bar_chart,
    skill_gap_chart, tier_donut_chart
)

st.set_page_config(
    page_title="Candidate Rankings", page_icon="🏆", layout="wide"
)
st.title("🏆 Candidate Rankings")

# ── Job validation ─────────────────────────────────────────────────────
if not st.session_state.get("selected_job_id"):
    st.warning("⚠️ Select a job from the sidebar to view rankings.")
    st.stop()

job_id    = st.session_state.selected_job_id
job_title = st.session_state.get("selected_job_title", "Selected Job")

st.subheader(f"Rankings for: {job_title}")

# ── Controls ───────────────────────────────────────────────────────────
col_refresh, col_rerank, col_filter = st.columns([1, 1, 3])

with col_refresh:
    if st.button("🔄 Refresh Rankings", use_container_width=True):
        st.cache_data.clear()

with col_rerank:
    if st.button("⚡ Re-Rank Now", use_container_width=True, type="primary"):
        with st.spinner("Ranking candidates..."):
            result = client.rank_candidates(job_id)
        if result["success"]:
            st.success("Rankings updated!")
        else:
            st.error(result["error"])

# ── Fetch ranking data ─────────────────────────────────────────────────
ranking_result = client.get_ranking(job_id)

if not ranking_result["success"]:
    st.warning(
        "No rankings available yet. "
        "Upload resumes and click 'Re-Rank Now'."
    )
    st.info(f"API response: {ranking_result.get('error', 'Unknown')}")
    st.stop()

ranking = ranking_result["data"]
candidates = ranking.get("ranked_candidates", [])

if not candidates:
    st.info("No candidates ranked yet. Upload and process resumes first.")
    st.stop()

# ── Summary metrics row ────────────────────────────────────────────────
total       = ranking.get("total_candidates", 0)
tier_dist   = ranking.get("tier_distribution", {})
score_stats = ranking.get("score_statistics", {})

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Candidates", total)
col2.metric("Priority (≥80%)",  tier_dist.get("priority", 0),
            delta="Interview now", delta_color="off")
col3.metric("Standard (≥65%)",  tier_dist.get("standard", 0),
            delta="Phone screen", delta_color="off")
col4.metric("Reserve (≥50%)",   tier_dist.get("reserve", 0))
col5.metric("Avg Score",
            f"{score_stats.get('mean_score', 0)*100:.1f}%")

st.divider()

# ── Tabbed views ───────────────────────────────────────────────────────
tab_list, tab_shortlist, tab_detail, tab_compare = st.tabs([
    "📋 Full Ranking",
    "⭐ Shortlists",
    "🔍 Candidate Detail",
    "📊 Pool Overview",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1: FULL RANKED LIST
# ══════════════════════════════════════════════════════════════════════
with tab_list:
    # Filters
    col_score, col_tier, col_search = st.columns([1, 1, 2])

    with col_score:
        min_score_filter = st.slider(
            "Min ATS Score (%)", 0, 100, 0, step=5
        )
    with col_tier:
        tier_filter = st.multiselect(
            "Score Tier",
            ["Excellent", "Good", "Fair", "Poor", "Very Poor"],
            default=["Excellent", "Good", "Fair", "Poor", "Very Poor"],
        )
    with col_search:
        search_term = st.text_input(
            "🔍 Search by name or skill",
            placeholder="e.g. 'John Smith' or 'python'"
        )

    # Build DataFrame
    rows = []
    for c in candidates:
        # Apply filters
        if c.get("ats_score_percent", 0) < min_score_filter:
            continue
        if c.get("score_tier") not in tier_filter:
            continue
        if search_term:
            name   = (c.get("candidate_name") or "").lower()
            skills = " ".join(c.get("top_skills", [])).lower()
            if search_term.lower() not in name and search_term.lower() not in skills:
                continue

        rows.append({
            "Rank":       c.get("rank"),
            "Percentile": f"{c.get('percentile', 0):.0f}th",
            "Name":       c.get("candidate_name") or "Unknown",
            "Score":      f"{c.get('ats_score_percent', 0):.1f}%",
            "Tier":       c.get("score_tier"),
            "Skill Match":f"{c.get('skill_match_rate', 0)*100:.0f}%",
            "Achievements":c.get("achievement_count", 0),
            "Skills":     c.get("total_skills_found", 0),
            "Shortlist":  c.get("shortlist_tier", "").upper(),
            "file_id":    c.get("file_id"),
        })

    if rows:
        df = pd.DataFrame(rows)
        display_df = df.drop(columns=["file_id"])

        # Color rows by tier
        def color_tier(val):
            colors = {
                "Excellent": "color: #155724; background-color: #d4edda",
                "Good":      "color: #004085; background-color: #cce5ff",
                "Fair":      "color: #856404; background-color: #fff3cd",
                "Poor":      "color: #721c24; background-color: #f8d7da",
            }
            return colors.get(val, "")

        styled = display_df.style.applymap(
            color_tier, subset=["Tier"]
        )

        st.dataframe(
            styled,
            use_container_width=True,
            height=500,
        )
        st.caption(f"Showing {len(rows)} of {total} candidates")

        # Export
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            f"rankings_{job_id[:8]}.csv",
            "text/csv",
        )
    else:
        st.info("No candidates match the current filters.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2: SHORTLISTS
# ══════════════════════════════════════════════════════════════════════
with tab_shortlist:
    shortlist_tier = st.radio(
        "Shortlist Tier",
        ["priority", "standard", "reserve"],
        format_func=lambda t: {
            "priority": "⭐ Priority — Immediate Interview (≥80%)",
            "standard": "✅ Standard — Phone Screen (≥65%)",
            "reserve":  "🔄 Reserve — Hold (≥50%)",
        }[t],
        horizontal=True,
    )

    shortlist_result = client.get_shortlist(job_id, tier=shortlist_tier)

    if shortlist_result["success"]:
        sl_data = shortlist_result["data"]
        sl_candidates = sl_data.get("candidates", [])

        st.metric(
            f"{shortlist_tier.title()} Shortlist",
            f"{sl_data.get('count', 0)} candidates"
        )

        if sl_candidates:
            for i, cand in enumerate(sl_candidates, 1):
                with st.container():
                    col_rank, col_info, col_score, col_skills = st.columns(
                        [1, 3, 2, 3]
                    )

                    with col_rank:
                        st.metric("Rank", f"#{cand.get('rank', i)}")
                        st.caption(f"{cand.get('percentile', 0):.0f}th pct")

                    with col_info:
                        st.markdown(
                            f"**{cand.get('candidate_name') or 'Unknown'}**"
                        )
                        st.caption(
                            cand.get("candidate_email") or
                            f"ID: {cand.get('file_id', '')[:8]}..."
                        )
                        st.caption(cand.get("recommendation", "")[:120] + "...")

                    with col_score:
                        score = cand.get("ats_score", 0)
                        st.metric(
                            "ATS Score",
                            f"{score*100:.1f}%",
                            cand.get("score_tier"),
                        )

                    with col_skills:
                        matched = cand.get("matched_skills", [])[:5]
                        missing = cand.get("missing_skills", [])[:3]
                        if matched:
                            st.markdown(
                                "✅ " +
                                " ".join(f"`{s}`" for s in matched)
                            )
                        if missing:
                            st.markdown(
                                "❌ " +
                                " ".join(f"`{s}`" for s in missing)
                            )

                    st.divider()
        else:
            st.info(f"No {shortlist_tier} candidates for this job.")
    else:
        st.error(shortlist_result.get("error", "Failed to load shortlist."))


# ══════════════════════════════════════════════════════════════════════
# TAB 3: CANDIDATE DETAIL
# ══════════════════════════════════════════════════════════════════════
with tab_detail:
    st.subheader("Candidate Deep-Dive")

    # Build name → file_id mapping for selectbox
    name_map = {
        f"#{c['rank']} — {c.get('candidate_name') or c['file_id'][:8]}": c["file_id"]
        for c in candidates
    }

    selected_name = st.selectbox(
        "Select Candidate",
        options=list(name_map.keys()),
        index=None,
        placeholder="Choose a candidate..."
    )

    if selected_name:
        selected_file_id = name_map[selected_name]
        # Find full candidate data
        cand_data = next(
            (c for c in candidates if c["file_id"] == selected_file_id),
            None
        )

        if cand_data:
            # ── Header ─────────────────────────────────────────────────
            col_gauge, col_meta = st.columns([1, 2])

            with col_gauge:
                st.plotly_chart(
                    score_gauge(cand_data.get("ats_score", 0)),
                    use_container_width=True
                )

            with col_meta:
                st.markdown(
                    f"### {cand_data.get('candidate_name') or 'Unknown Candidate'}"
                )
                st.caption(cand_data.get("candidate_email", "No email"))

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Rank",
                              f"#{cand_data['rank']} of {cand_data['total_candidates']}")
                col_r2.metric("Percentile",
                              f"{cand_data.get('percentile', 0):.0f}th")
                col_r3.metric("Skill Match",
                              f"{cand_data.get('skill_match_rate', 0)*100:.0f}%")

                tier = cand_data.get("score_tier", "N/A")
                tier_icons = {
                    "Excellent": "🟢", "Good": "🔵",
                    "Fair": "🟡", "Poor": "🔴"
                }
                st.markdown(
                    f"**Tier:** {tier_icons.get(tier, '⚪')} {tier} | "
                    f"**Shortlist:** {cand_data.get('shortlist_tier', 'N/A').upper()}"
                )

                st.info(cand_data.get("recommendation", ""))

            st.divider()

            # ── Score components ────────────────────────────────────────
            criteria = cand_data.get("rank_criteria", [])
            if criteria:
                st.subheader("Ranking Criteria Breakdown")
                criteria_data = [
                    {
                        "name":         cr["name"].replace("_", " ").title(),
                        "raw_score":    cr["raw_value"],
                        "weight":       cr["weight"],
                        "weighted_contribution": cr["contribution"],
                    }
                    for cr in criteria
                ]
                st.plotly_chart(
                    component_bar_chart(criteria_data),
                    use_container_width=True
                )

            # ── Skill gap ───────────────────────────────────────────────
            skill_gap = {
                "matched":    cand_data.get("matched_skills", []),
                "missing":    cand_data.get("missing_skills", []),
                "extra":      cand_data.get("extra_skills", []),
                "match_rate": cand_data.get("skill_match_rate", 0),
            }

            st.subheader("Skill Analysis")
            col_m, col_gap, col_e = st.columns(3)
            col_m.metric("Matched",  len(skill_gap["matched"]))
            col_gap.metric("Missing", len(skill_gap["missing"]))
            col_e.metric("Extra",    len(skill_gap["extra"]))

            st.plotly_chart(
                skill_gap_chart(skill_gap),
                use_container_width=True
            )


# ══════════════════════════════════════════════════════════════════════
# TAB 4: POOL OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Candidate Pool Overview")

    col_donut, col_stats = st.columns([1, 1])

    with col_donut:
        st.plotly_chart(
            tier_donut_chart(tier_dist),
            use_container_width=True
        )

    with col_stats:
        st.subheader("Pool Statistics")
        if score_stats:
            stat_rows = [
                ("Mean Score",   f"{score_stats.get('mean_score', 0)*100:.1f}%"),
                ("Median Score", f"{score_stats.get('median_score', 0)*100:.1f}%"),
                ("Std Deviation",f"{score_stats.get('std_score', 0)*100:.1f}%"),
                ("Min Score",    f"{score_stats.get('min_score', 0)*100:.1f}%"),
                ("Max Score",    f"{score_stats.get('max_score', 0)*100:.1f}%"),
                ("P25 Score",    f"{score_stats.get('p25_score', 0)*100:.1f}%"),
                ("P75 Score",    f"{score_stats.get('p75_score', 0)*100:.1f}%"),
                ("Above Threshold", score_stats.get("total_above_threshold", 0)),
            ]
            stat_df = pd.DataFrame(stat_rows, columns=["Metric", "Value"])
            st.dataframe(stat_df, use_container_width=True, hide_index=True)
