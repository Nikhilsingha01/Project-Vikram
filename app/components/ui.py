"""Reusable UI rendering helpers for the FLUX Streamlit app."""

from typing import Any, Dict, Iterable, Optional

import streamlit as st


def display_processing_status(steps: Optional[Iterable[Dict[str, str]]]) -> None:
    """Render pipeline stages and their current state."""
    st.subheader("Processing Pipeline")
    if not steps:
        st.info("Processing has not started.")
        return
    for step in steps:
        status = step.get("status", "unknown")
        icon = {"completed": "✅", "not_implemented": "○", "failed": "❌"}.get(
            status, "•"
        )
        detail = step.get("detail")
        label = f"{icon} {step.get('name', 'Unnamed stage')}"
        st.write(label)
        if detail:
            st.caption(detail)


def display_image_comparison(
    source_image: Any,
    reference_image: Any,
    registered_image: Any = None,
) -> None:
    """Render available source, reference, and registered images."""
    st.subheader("Images")
    columns = st.columns(3 if registered_image is not None else 2)
    with columns[0]:
        st.image(source_image, caption="Source Image", use_container_width=True)
    with columns[1]:
        st.image(reference_image, caption="Reference Image", use_container_width=True)
    if registered_image is not None:
        with columns[2]:
            st.image(
                registered_image,
                caption="Registered Image",
                use_container_width=True,
            )
    else:
        st.info("A registered image will appear here after real registration is connected.")


def display_metrics(result: Dict[str, Any]) -> None:
    """Render registration metrics, including unavailable values safely."""
    st.subheader("Registration Metrics")
    metrics = (
        ("RMSE", result.get("rmse")),
        ("Mean Error", result.get("mean_error")),
        ("Maximum Error", result.get("max_error")),
        ("Matches", result.get("matches")),
        ("Inliers", result.get("inliers")),
        ("Inlier Ratio", result.get("inlier_ratio")),
        ("Confidence", result.get("confidence")),
    )
    columns = st.columns(4)
    for index, (label, value) in enumerate(metrics):
        with columns[index % len(columns)]:
            display_value = "Unavailable" if value is None else value
            st.metric(label, display_value)


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
