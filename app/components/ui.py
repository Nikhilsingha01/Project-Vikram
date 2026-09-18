"""Reusable UI rendering helpers for the FLUX Streamlit app."""

from typing import Any, Dict, Iterable, Optional

import streamlit as st


def display_processing_status(steps: Optional[Iterable[Dict[str, str]]]) -> None:
    """Render user-facing registration stages and their current state."""
    st.subheader("Registration Processing")
    if not steps:
        st.info("Processing has not started.")
        return
    stage_names = {
        "M1": "Candidate Search",
        "M2": "Feature Matching",
        "M3": "Geometric Verification",
        "M4": "Sub-pixel Refinement",
    }
    for step in steps:
        name = step.get("name", "").upper()
        if name == "PIPELINE":
            continue
        status = step.get("status", "unknown")
        icon = {"completed": "✓", "not_implemented": "○", "failed": "✕"}.get(status, "•")
        detail = step.get("detail")
        label = f"{icon} {stage_names.get(name, step.get('name', 'Unnamed stage'))}"
        with st.container(border=True):
            st.markdown(f"<div style='display:flex; align-items:center; gap:0.7rem; color:#eef6ff; font-weight:600; padding:0.1rem 0;'> <span style='color:#67e8f9; font-size:1.05rem;'>{icon}</span> <span>{label}</span></div>", unsafe_allow_html=True)
            if detail:
                st.caption(detail)


def display_image_comparison(
    source_image: Any,
    reference_image: Any,
    registered_image: Any = None,
) -> None:
    """Render available source, reference, and registered images."""
    st.subheader("Image Comparison")
    columns = st.columns(3 if registered_image is not None else 2)
    with columns[0]:
        st.image(source_image, caption="Source Image", width="stretch")
    with columns[1]:
        st.image(reference_image, caption="Reference Image", width="stretch")
    if registered_image is not None:
        with columns[2]:
            st.image(
                registered_image,
                caption="Registered Image",
                width="stretch",
            )


def display_metrics(result: Dict[str, Any]) -> None:
    """Render the registration metrics using real pipeline values."""
    st.subheader("Registration Metrics")
    matches = result.get("matches")
    inliers = result.get("inliers")
    inlier_ratio = result.get("inlier_ratio")
    if inlier_ratio is None and matches:
        inlier_ratio = float(inliers) / float(matches)

    def format_count(value: Any) -> str:
        return "N/A" if value is None else f"{int(value)}"

    def format_percent(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.1%}"

    def format_error(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2f} px"

    with st.container(border=True):
        first_row = st.columns(2)
        with first_row[0]:
            st.metric("Feature Matches", format_count(matches))
        with first_row[1]:
            st.metric("Verified Inliers", format_count(inliers))

        second_row = st.columns(2)
        with second_row[0]:
            st.metric("Inlier Ratio", format_percent(inlier_ratio))
        with second_row[1]:
            st.metric("Confidence", format_percent(result.get("confidence")))

        third_row = st.columns(2)
        with third_row[0]:
            st.metric("RMSE", format_error(result.get("rmse")))
        with third_row[1]:
            st.metric("Mean Error", format_error(result.get("mean_error")))

        st.metric("Maximum Error", format_error(result.get("max_error")))


def display_match_status(result: Dict[str, Any]) -> None:
    """Render acceptance state and a user-facing failure reason."""
    status = str(result.get("status", "unknown")).lower()
    if status == "accepted":
        st.success("MATCH ACCEPTED")
    elif status == "rejected":
        st.warning("MATCH REJECTED")
    else:
        st.info(f"MATCH STATUS: {status.upper()}")
    if result.get("failure_reason"):
        st.caption(str(result["failure_reason"]))
    if result.get("demo_mode"):
        st.info("DEMO MODE: no scientifically valid localization or registration was performed.")
