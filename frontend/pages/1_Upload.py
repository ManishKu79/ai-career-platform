

import streamlit as st
import pandas as pd
from components.api_client import client
from components.charts import score_gauge, skill_gap_chart, radar_chart

st.set_page_config(page_title="Upload Resume", page_icon="📄", layout="wide")
st.title("📄 Resume Upload & Processing")

# ── Job validation ─────────────────────────────────────────────────────
if not st.session_state.get("selected_job_id"):
    st.warning("⚠️ Select a job from the sidebar before uploading resumes.")
    st.stop()

job_id    = st.session_state.selected_job_id
job_title = st.session_state.get("selected_job_title", "Selected Job")

st.info(f"📋 Scoring resumes against: **{job_title}**")

# ── Upload Mode Toggle ────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Upload", "Batch Upload"])


# ══════════════════════════════════════════════════════════════════════
# SINGLE UPLOAD TAB
# ══════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Upload a Single Resume")

    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX resume",
        type=["pdf", "docx"],
        help="Maximum file size: 10MB",
    )

    col_options, col_info = st.columns([1, 2])
    with col_options:
        run_ranking = st.checkbox(
            "Re-rank candidates after upload",
            value=True,
            help="Updates the ranking for the selected job after scoring"
        )

    if uploaded_file and st.button("🚀 Process Resume", type="primary", use_container_width=True):

        # ── Progress display ───────────────────────────────────────────
        progress_bar = st.progress(0, text="Initializing pipeline...")
        status_area  = st.empty()

        with st.spinner("Processing resume through full pipeline..."):

            # Update progress at each stage
            stage_messages = [
                (10,  "Stage 1/5: Parsing PDF/DOCX..."),
                (30,  "Stage 2/5: Running NLP pipeline..."),
                (55,  "Stage 3/5: Extracting skills..."),
                (75,  "Stage 4/5: Persisting to MongoDB..."),
                (90,  "Stage 5/5: Computing ATS score..."),
                (100, "Pipeline complete!"),
            ]

            # Show animated progress (simulated stages)
            import time
            for pct, msg in stage_messages[:-1]:
                progress_bar.progress(pct, text=msg)
                time.sleep(0.3)

            # Actual API call
            result = client.process_pipeline(
                file_bytes=uploaded_file.read(),
                filename=uploaded_file.name,
                job_id=job_id,
                run_ranking=run_ranking,
            )

            progress_bar.progress(100, text="Pipeline complete!")

        # ── Display results ────────────────────────────────────────────
        if result["success"]:
            data = result["data"]
            st.session_state.last_pipeline_result = data

            # Store file_id for later use
            file_id = data.get("file_id")
            if file_id not in st.session_state.uploaded_file_ids:
                st.session_state.uploaded_file_ids.append(file_id)

            st.success(f"✅ Resume processed successfully! File ID: `{file_id[:8]}...`")

            # ── Score summary at top ───────────────────────────────────
            ats = data.get("ats_score", {})
            if ats:
                st.subheader("ATS Score")
                col_gauge, col_details = st.columns([1, 2])

                with col_gauge:
                    score  = ats.get("final_score", 0)
                    st.plotly_chart(
                        score_gauge(score, "ATS Score"),
                        use_container_width=True
                    )

                with col_details:
                    tier   = ats.get("score_tier", "N/A")
                    passes = ats.get("passes_threshold", False)

                    tier_color = {
                        "Excellent": "🟢", "Good": "🔵",
                        "Fair": "🟡", "Poor": "🔴", "Very Poor": "⛔"
                    }
                    icon = tier_color.get(tier, "⚪")

                    st.metric(
                        "Score Tier",
                        f"{icon} {tier}",
                        delta="Passes threshold" if passes else "Below threshold",
                        delta_color="normal" if passes else "inverse",
                    )

                    st.markdown(f"**Recommendation:**\n> {ats.get('recommendation', '')}")

                    matched_kw = ats.get("matched_keywords", [])
                    if matched_kw:
                        st.markdown(
                            f"**Matched Keywords:** "
                            + " ".join([f"`{k}`" for k in matched_kw[:8]])
                        )

            # ── Detailed results in expanders ──────────────────────────
            with st.expander("📑 Parse Results", expanded=False):
                parse = data.get("parse_result", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Candidate",  parse.get("candidate_name", "N/A"))
                col2.metric("Word Count", parse.get("word_count", 0))
                col3.metric("Pages",      parse.get("page_count", "N/A"))
                st.caption(f"Email: {parse.get('candidate_email', 'N/A')}")

            with st.expander("🔤 NLP Summary", expanded=False):
                nlp = data.get("nlp_summary", {})
                cols = st.columns(4)
                cols[0].metric("Total Tokens",    nlp.get("total_tokens", 0))
                cols[1].metric("Unique Tokens",   nlp.get("unique_tokens", 0))
                cols[2].metric("Entities Found",  nlp.get("entities_found", 0))
                cols[3].metric("Achievements",    nlp.get("achievement_count", 0))

                if nlp.get("top_words"):
                    st.markdown("**Top Words:**")
                    words_df = pd.DataFrame(
                        nlp["top_words"], columns=["Word", "Count"]
                    )
                    st.dataframe(words_df, use_container_width=True, height=150)

            with st.expander("🔧 Extracted Skills", expanded=True):
                skills = data.get("skill_summary", {})
                st.metric("Total Skills Found", skills.get("total_skills_found", 0))

                top_skills = skills.get("top_skills", [])
                if top_skills:
                    st.markdown("**Top Skills:**")
                    st.markdown(" ".join([f"`{s}`" for s in top_skills]))

                by_cat = skills.get("skills_by_category", {})
                if by_cat:
                    st.plotly_chart(
                        radar_chart(by_cat),
                        use_container_width=True
                    )

            with st.expander("📊 Skill Gap Analysis", expanded=True):
                if ats and ats.get("skill_gap"):
                    gap = ats["skill_gap"]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Matched Skills",  len(gap.get("matched", [])))
                    col2.metric("Missing Skills",  len(gap.get("missing", [])))
                    col3.metric("Match Rate",
                                f"{gap.get('match_rate', 0)*100:.0f}%")

                    st.plotly_chart(
                        skill_gap_chart(gap),
                        use_container_width=True
                    )

                    if gap.get("missing"):
                        st.error(
                            "**Missing required skills:** " +
                            ", ".join(f"`{s}`" for s in gap["missing"][:10])
                        )

            # Timing info
            st.caption(
                f"Pipeline completed in {data.get('total_ms', 0):.0f}ms | "
                f"File ID: `{data.get('file_id', 'N/A')}`"
            )

        else:
            st.error(f"❌ Pipeline failed: {result['error']}")
            progress_bar.empty()


# ══════════════════════════════════════════════════════════════════════
# BATCH UPLOAD TAB
# ══════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch Upload — Multiple Resumes")
    st.caption("Upload up to 20 resumes at once for bulk processing.")

    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX resume files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Maximum 20 files, 10MB each",
    )

    if uploaded_files:
        st.info(f"**{len(uploaded_files)} file(s) selected**")

        if len(uploaded_files) > 20:
            st.error("Maximum 20 files per batch. Please reduce selection.")
        else:
            if st.button("🚀 Process All Resumes", type="primary", use_container_width=True):

                files_data = [
                    (f.name, f.read()) for f in uploaded_files
                ]

                with st.spinner(f"Processing {len(files_data)} resumes..."):
                    result = client.batch_process(files_data, job_id)

                if result["success"]:
                    data = result["data"]

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Files",  data.get("total_files", 0))
                    col2.metric("Successful",   data.get("successful", 0))
                    col3.metric("Failed",       data.get("failed", 0))
                    col4.metric("Time (ms)",    data.get("total_ms", 0))

                    # Results table
                    results_list = data.get("results", [])
                    if results_list:
                        st.subheader("Processing Results")
                        df = pd.DataFrame(results_list)

                        # Color code score tier
                        def highlight_tier(row):
                            colors = {
                                "Excellent": "background-color: #d4edda",
                                "Good":      "background-color: #cce5ff",
                                "Fair":      "background-color: #fff3cd",
                                "Poor":      "background-color: #f8d7da",
                            }
                            color = colors.get(row.get("score_tier", ""), "")
                            return [color] * len(row)

                        styled_df = df[[
                            "filename", "candidate_name",
                            "ats_score", "score_tier", "skills_found"
                        ]].style.apply(highlight_tier, axis=1)

                        st.dataframe(styled_df, use_container_width=True)

                    # Error details
                    if data.get("errors"):
                        with st.expander(f"⚠️ {data['failed']} Failed Files"):
                            for err in data["errors"]:
                                st.error(f"**{err['filename']}**: {err['error']}")
                else:
                    st.error(f"❌ Batch processing failed: {result['error']}")
