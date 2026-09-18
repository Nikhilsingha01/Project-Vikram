"""Streamlit entry point for the FLUX demonstration application."""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.components import (
    display_image_comparison,
    display_match_status,
    display_metrics,
    display_processing_status,
)
from app.pipeline import run_pipeline


SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE_IMAGES = {
    "Chandrayaan-2 source": PROJECT_ROOT / "data/raw/lunar_reference/LRO/quickmap-lroc.png",
    "LRO reference": PROJECT_ROOT / "data/prepared/lunar_reference/LRO_gray.png",
}


def main() -> None:
    """Render the FLUX application."""
    st.set_page_config(
        page_title="FLUX | Lunar Image Registration",
        page_icon="🌙",
        layout="wide",
    )
    st.title("FLUX")
    st.caption("Lunar Image Registration & Localization")
    st.success("This app runs the real FLUX registration pipeline on the selected images.")

    source_upload, reference_upload = st.columns(2)
    with source_upload:
        source_mode = st.radio(
            "Source input",
            ["Prototype image", "Upload image"],
            horizontal=True,
            key="source_mode",
        )
        source_prototype = None
        if source_mode == "Prototype image":
            source_prototype = st.selectbox(
                "Source prototype",
                list(PROTOTYPE_IMAGES),
                key="source_prototype",
            )
        source_file = st.file_uploader(
            "Source image",
            type=SUPPORTED_TYPES,
            key="source_image",
            disabled=source_mode == "Prototype image",
        )
    with reference_upload:
        reference_mode = st.radio(
            "Reference input",
            ["Prototype image", "Upload image"],
            horizontal=True,
            key="reference_mode",
        )
        reference_prototype = None
        if reference_mode == "Prototype image":
            reference_prototype = st.selectbox(
                "Reference prototype",
                list(PROTOTYPE_IMAGES),
                index=1,
                key="reference_prototype",
            )
        reference_file = st.file_uploader(
            "Reference image",
            type=SUPPORTED_TYPES,
            key="reference_image",
            disabled=reference_mode == "Prototype image",
        )

    source_image = _read_selected_image(
        source_file,
        PROTOTYPE_IMAGES[source_prototype] if source_prototype else None,
        "source",
    )
    reference_image = _read_selected_image(
        reference_file,
        PROTOTYPE_IMAGES[reference_prototype] if reference_prototype else None,
        "reference",
    )
    if source_image is not None and reference_image is not None:
        display_image_comparison(source_image, reference_image)

    if st.button("Run FLUX", type="primary", width="stretch"):
        if source_image is None or reference_image is None:
            st.error("Select or upload both a source image and a reference image before running FLUX.")
            return
        try:
            result = run_pipeline(source_image, reference_image)
        except (TypeError, ValueError) as error:
            st.error(f"Unable to process the uploaded images: {error}")
            return
        _display_result(result, source_image, reference_image)


def _read_selected_image(
    uploaded_file: Optional[Any], prototype_path: Optional[Path], label: str
) -> Optional[np.ndarray]:
    """Read either a repository prototype or an uploaded image."""
    try:
        if prototype_path is not None:
            image = Image.open(prototype_path).convert("RGB")
        elif uploaded_file is not None:
            image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        else:
            return None
    except (UnidentifiedImageError, OSError, ValueError):
        st.error(f"The selected {label} image is not a valid supported image.")
        return None
    return np.asarray(image)


def _display_result(
    result: Any, source_image: np.ndarray, reference_image: np.ndarray
) -> None:
    """Render a standard pipeline result."""
    result_data = result.to_dict() if hasattr(result, "to_dict") else result
    matches = result_data.get("final_matches")
    inliers = result_data.get("final_inliers")
    metrics_data = {
        **result_data,
        "matches": matches,
        "inliers": inliers,
        "inlier_ratio": result_data.get("final_inlier_ratio"),
        "rmse": result_data.get("final_rmse"),
        "mean_error": result_data.get("final_mean_error"),
        "max_error": result_data.get("final_max_error"),
        "confidence": result_data.get("final_confidence"),
    }
    st.divider()
    st.header("Result")
    display_match_status(result_data)
    display_metrics(metrics_data)
    display_processing_status(result_data.get("processing_steps"))
    if result_data.get("total_processing_time_ms") is not None:
        with st.expander("Processing details"):
            st.metric("Total processing time", f"{result_data['total_processing_time_ms']} ms")
            stage_timings = {
                name.upper(): details.get("details", {}).get("duration_ms")
                for name, details in result_data.get("stage_results", {}).items()
                if details.get("details", {}).get("duration_ms") is not None
            }
            if stage_timings:
                st.caption(
                    "Stage timings: "
                    + " | ".join(f"{name} {duration} ms" for name, duration in stage_timings.items())
                )
    display_image_comparison(
        source_image,
        reference_image,
        getattr(result, "registered_image", result_data.get("registered_image")),
    )


if __name__ == "__main__":
    main()