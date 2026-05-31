# frontend/components/charts.py

import plotly.express      as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional

# Brand color palette used across all charts
COLORS = {
    "primary":   "#1f77b4",
    "success":   "#2ca02c",
    "warning":   "#ff7f0e",
    "danger":    "#d62728",
    "purple":    "#9467bd",
    "teal":      "#17becf",
    "gray":      "#7f7f7f",
    "excellent": "#2ca02c",
    "good":      "#1f77b4",
    "fair":      "#ff7f0e",
    "poor":      "#d62728",
}

TIER_COLORS = {
    "Excellent": COLORS["excellent"],
    "Good":      COLORS["good"],
    "Fair":      COLORS["warning"],
    "Poor":      COLORS["danger"],
    "Very Poor": "#8B0000",
}


def score_gauge(score: float, title: str = "ATS Score") -> go.Figure:
    """
    Creates a gauge chart displaying a single ATS score.

    A gauge is the most intuitive way to display a single score —
    recruiters immediately understand "75 out of 100".

    The gauge has color-coded zones matching our tier thresholds:
    Red (0-50) → Orange (50-65) → Blue (65-80) → Green (80-100)

    Args:
        score: ATS score (0.0 to 1.0)
        title: Chart title

    Returns:
        Plotly Figure with gauge chart
    """
    score_pct = round(score * 100, 1)

    # Determine color based on tier
    if score >= 0.80:
        color = COLORS["excellent"]
    elif score >= 0.65:
        color = COLORS["good"]
    elif score >= 0.50:
        color = COLORS["warning"]
    else:
        color = COLORS["danger"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score_pct,

        # Delta shows difference from threshold (50%)
        delta={
            "reference": 50,
            "increasing": {"color": COLORS["success"]},
            "decreasing": {"color": COLORS["danger"]},
        },

        # Number display formatting
        number={"suffix": "%", "font": {"size": 48}},

        # Gauge configuration
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "darkblue",
                "tickvals": [0, 50, 65, 80, 100],
            },
            "bar": {"color": color, "thickness": 0.3},

            # Color-coded background zones
            "steps": [
                {"range": [0,  50],  "color": "#ffebee"},  # Light red
                {"range": [50, 65],  "color": "#fff3e0"},  # Light orange
                {"range": [65, 80],  "color": "#e3f2fd"},  # Light blue
                {"range": [80, 100], "color": "#e8f5e9"},  # Light green
            ],

            # Threshold line at our minimum passing score
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": title, "font": {"size": 20}},
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
    )

    return fig


def component_bar_chart(component_scores: List[Dict]) -> go.Figure:
    """
    Horizontal bar chart showing ATS score component breakdown.

    Visualizes the 5 scoring components side-by-side so recruiters
    can see exactly which dimensions drove the overall score.

    Args:
        component_scores: List of ComponentScore dicts from API

    Returns:
        Horizontal bar chart Figure
    """
    if not component_scores:
        return go.Figure()

    names         = []
    raw_scores    = []
    contributions = []
    colors_list   = []

    for comp in component_scores:
        # Clean up name for display: "tfidf_similarity" → "TF-IDF Similarity"
        display_name = (
            comp["name"]
            .replace("_", " ")
            .title()
            .replace("Tfidf", "TF-IDF")
        )
        names.append(display_name)
        raw_scores.append(round(comp["raw_score"] * 100, 1))
        contributions.append(round(comp["weighted_contribution"] * 100, 1))

        # Color based on raw score
        if comp["raw_score"] >= 0.70:
            colors_list.append(COLORS["success"])
        elif comp["raw_score"] >= 0.50:
            colors_list.append(COLORS["good"])
        else:
            colors_list.append(COLORS["danger"])

    fig = go.Figure()

    # Raw score bars
    fig.add_trace(go.Bar(
        name="Raw Score (%)",
        y=names,
        x=raw_scores,
        orientation="h",
        marker_color=colors_list,
        text=[f"{s:.1f}%" for s in raw_scores],
        textposition="inside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Score Component Breakdown",
        xaxis_title="Score (%)",
        xaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )

    # Add vertical line at 50% threshold
    fig.add_vline(
        x=50,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        annotation_position="top",
    )

    return fig


def skill_gap_chart(skill_gap: Dict, max_skills: int = 15) -> go.Figure:
    """
    Grouped bar chart showing matched vs missing skills.

    Two bars per skill category:
    - Green bar: skills candidate HAS that job requires
    - Red bar:   skills job requires that candidate LACKS

    Args:
        skill_gap: Skill gap dict with matched/missing lists
        max_skills: Maximum skills to display

    Returns:
        Grouped bar chart Figure
    """
    matched = skill_gap.get("matched", [])[:max_skills]
    missing = skill_gap.get("missing", [])[:max_skills]

    # Combine all skills for x-axis
    all_skills = list(set(matched + missing))
    if not all_skills:
        return go.Figure()

    matched_set = set(matched)
    missing_set = set(missing)

    fig = go.Figure()

    # Matched skills (green)
    matched_vals = [1 if s in matched_set else 0 for s in all_skills]
    fig.add_trace(go.Bar(
        name="✓ Has Skill",
        x=all_skills,
        y=matched_vals,
        marker_color=COLORS["success"],
        opacity=0.85,
    ))

    # Missing skills (red)
    missing_vals = [1 if s in missing_set else 0 for s in all_skills]
    fig.add_trace(go.Bar(
        name="✗ Missing",
        x=all_skills,
        y=missing_vals,
        marker_color=COLORS["danger"],
        opacity=0.85,
    ))

    match_rate = skill_gap.get("match_rate", 0)
    fig.update_layout(
        title=f"Skill Gap Analysis — {match_rate*100:.0f}% Match Rate",
        xaxis_title="Required Skills",
        yaxis_title="",
        yaxis=dict(showticklabels=False, range=[0, 1.5]),
        barmode="group",
        height=350,
        margin=dict(l=20, r=20, t=60, b=100),
        paper_bgcolor="white",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def score_histogram(histogram_data: List[Dict], mean_score: float) -> go.Figure:
    """
    Histogram of ATS score distribution across candidate pool.

    Shows recruiter the quality distribution: "Most candidates
    scored between 40-60% — this is a competitive position."

    Args:
        histogram_data: List of bin dicts from aggregation API
        mean_score:     Mean score for annotation line

    Returns:
        Bar chart Figure styled as histogram
    """
    if not histogram_data:
        return go.Figure()

    labels = [b["label"] for b in histogram_data]
    counts = [b["count"]  for b in histogram_data]

    # Color bins by score range (below threshold = red, above = blue/green)
    bin_colors = []
    for b in histogram_data:
        mid = (b["bin_start"] + b["bin_end"]) / 2
        if mid >= 0.80:
            bin_colors.append(COLORS["excellent"])
        elif mid >= 0.65:
            bin_colors.append(COLORS["good"])
        elif mid >= 0.50:
            bin_colors.append(COLORS["warning"])
        else:
            bin_colors.append(COLORS["danger"])

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        marker_color=bin_colors,
        text=counts,
        textposition="outside",
        hovertemplate="Score range: %{x}<br>Candidates: %{y}<extra></extra>",
    ))

    # Add mean score annotation
    mean_bin = f"{int(mean_score*100//10*10)}-{int(mean_score*100//10*10+10)}%"
    fig.add_vline(
        x=mean_bin,
        line_dash="dot",
        line_color="navy",
        annotation_text=f"Mean: {mean_score*100:.1f}%",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Score Distribution",
        xaxis_title="ATS Score Range",
        yaxis_title="Number of Candidates",
        height=350,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


def tier_donut_chart(tier_distribution: Dict) -> go.Figure:
    """
    Donut chart showing candidate tier distribution.

    Quick visual of: how many Excellent / Good / Fair / Poor candidates.

    Args:
        tier_distribution: Dict mapping tier_name → count

    Returns:
        Donut pie chart Figure
    """
    # Map API tier names to display names
    tier_map = {
        "priority": "Priority (≥80%)",
        "standard": "Standard (≥65%)",
        "reserve":  "Reserve (≥50%)",
        "reject":   "Reject (<50%)",
        "Excellent":"Excellent (≥80%)",
        "Good":     "Good (≥65%)",
        "Fair":     "Fair (≥50%)",
        "Poor":     "Poor (<50%)",
        "Very Poor":"Very Poor",
    }

    labels = []
    values = []
    colors_list = []

    tier_color_map = {
        "priority": COLORS["excellent"],
        "standard": COLORS["good"],
        "reserve":  COLORS["warning"],
        "reject":   COLORS["danger"],
        "Excellent":COLORS["excellent"],
        "Good":     COLORS["good"],
        "Fair":     COLORS["warning"],
        "Poor":     COLORS["danger"],
        "Very Poor":COLORS["danger"],
    }

    for tier, count in tier_distribution.items():
        if count > 0:
            labels.append(tier_map.get(tier, tier))
            values.append(count)
            colors_list.append(tier_color_map.get(tier, COLORS["gray"]))

    if not values:
        return go.Figure()

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors_list,
        textinfo="percent+label",
        hovertemplate="%{label}: %{value} candidates (%{percent})<extra></extra>",
    ))

    total = sum(values)
    fig.update_layout(
        title="Candidate Tier Distribution",
        annotations=[{
            "text": f"{total}<br>Total",
            "x": 0.5, "y": 0.5,
            "font_size": 18,
            "showarrow": False,
        }],
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    return fig


def top_skills_bar(skills_data: List[Dict]) -> go.Figure:
    """
    Horizontal bar chart of most common skills across all resumes.

    Args:
        skills_data: List of {skill, count, percentage} dicts

    Returns:
        Horizontal bar chart Figure
    """
    if not skills_data:
        return go.Figure()

    df = pd.DataFrame(skills_data).head(20)
    df = df.sort_values("count", ascending=True)

    fig = px.bar(
        df,
        x="count",
        y="skill",
        orientation="h",
        color="count",
        color_continuous_scale="Blues",
        text="count",
        title="Top Skills Across All Resumes",
        labels={"count": "Occurrences", "skill": "Skill"},
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=500,
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor="white",
        xaxis_title="Number of Resumes",
        yaxis_title="",
        margin=dict(l=20, r=60, t=60, b=20),
    )

    return fig


def funnel_chart(funnel_data: List[Dict]) -> go.Figure:
    """
    Funnel chart showing candidate pipeline stages.

    Args:
        funnel_data: List of {stage, count, percentage} dicts

    Returns:
        Funnel chart Figure
    """
    if not funnel_data:
        return go.Figure()

    stages = [s["stage"]      for s in funnel_data]
    counts = [s["count"]      for s in funnel_data]
    pcts   = [s["percentage"] for s in funnel_data]

    fig = go.Figure(go.Funnel(
        y=stages,
        x=counts,
        textinfo="value+percent initial",
        textposition="inside",
        marker=dict(
            color=[
                COLORS["primary"],
                COLORS["teal"],
                COLORS["success"],
                COLORS["warning"],
                COLORS["purple"],
            ][:len(stages)]
        ),
    ))

    fig.update_layout(
        title="Candidate Processing Pipeline",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
    )

    return fig


def radar_chart(skills_by_category: Dict[str, int]) -> go.Figure:
    """
    Radar chart showing candidate's skill coverage by taxonomy category.

    Args:
        skills_by_category: Dict mapping category → skill count

    Returns:
        Radar (polar) chart Figure
    """
    if not skills_by_category:
        return go.Figure()

    # Clean category names for display
    category_display = {
        "programming_languages":    "Languages",
        "frameworks_and_libraries": "Frameworks",
        "data_science_and_ml":      "Data/ML",
        "databases":                "Databases",
        "cloud_and_devops":         "Cloud/DevOps",
        "engineering_practices":    "Practices",
    }

    categories = []
    counts     = []

    for cat, count in skills_by_category.items():
        categories.append(category_display.get(cat, cat))
        counts.append(count)

    # Close the radar polygon
    categories.append(categories[0])
    counts.append(counts[0])

    fig = go.Figure(go.Scatterpolar(
        r=counts,
        theta=categories,
        fill="toself",
        fillcolor=f"rgba(31, 119, 180, 0.3)",
        line_color=COLORS["primary"],
        name="Skill Coverage",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(counts) + 1],
            )
        ),
        title="Skill Category Coverage",
        height=350,
        showlegend=False,
        paper_bgcolor="white",
    )

    return fig