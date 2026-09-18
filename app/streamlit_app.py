import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repository root is on sys.path for Streamlit Community Cloud and local runs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    from app.components import (
        display_image_comparison,
        display_match_status,
        display_metrics,
        display_processing_status,
    )
    from app.pipeline import run_pipeline
except ModuleNotFoundError:
    from components import (  # type: ignore
        display_image_comparison,
        display_match_status,
        display_metrics,
        display_processing_status,
    )
    from pipeline import run_pipeline  # type: ignore


SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
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

    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #050b14 0%, #0a1220 45%, #071019 100%);
            }
            .main .block-container {
                max-width: 1420px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            h1 {
                margin-bottom: 0.15rem;
                letter-spacing: 0.08em;
                font-weight: 800;
                color: #f4f8ff;
                text-transform: uppercase;
            }
            .stCaption {
                color: #9fb6d0 !important;
                font-size: 0.92rem !important;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin-bottom: 1.2rem !important;
            }
            .stAlert, .stSuccess, .stWarning, .stInfo {
                border-radius: 16px;
                border: 1px solid rgba(96, 165, 250, 0.2);
                background: rgba(15, 23, 42, 0.8);
                box-shadow: 0 0 0 1px rgba(148, 163, 184, 0.08), 0 18px 30px rgba(2, 6, 23, 0.28);
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(12, 18, 28, 0.78);
                border: 1px solid rgba(96, 165, 250, 0.18);
                border-radius: 18px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 12px 28px rgba(2,6,23,0.22);
                padding: 1rem 1rem 0.35rem 1rem;
            }
            [data-testid="stFileUploaderDropzone"] {
                background: rgba(15, 23, 42, 0.72);
                border: 1px dashed rgba(96, 165, 250, 0.38);
                border-radius: 16px;
            }
            [data-testid="stFileUploaderDropzoneHover"] {
                border-color: rgba(34, 211, 238, 0.8);
                box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.25);
            }
            [data-testid="stRadio"] label,
            [data-testid="stSelectbox"] label,
            [data-testid="stFileUploader"] label,
            [data-testid="stExpander"] summary {
                color: #dfeaf6 !important;
                font-weight: 600;
            }
            [data-testid="stSelectbox"] > div,
            [data-testid="baseButton-secondary"],
            [data-testid="stHorizontalBlock"] {
                border-radius: 12px;
            }
            div[data-testid="stMetric"] {
                background: rgba(11, 18, 29, 0.85);
                border: 1px solid rgba(96, 165, 250, 0.18);
                border-radius: 16px;
                padding: 0.7rem 0.8rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            }
            div[data-testid="stMetric"] label {
                color: #9fb6d0 !important;
                font-weight: 600;
            }
            div[data-testid="stMetric"] div {
                color: #f8fbff !important;
                font-weight: 800;
            }
            [data-testid="stImage"] {
                border-radius: 18px;
                border: 1px solid rgba(96, 165, 250, 0.18);
                box-shadow: 0 14px 26px rgba(2, 6, 23, 0.28);
            }
            .stButton > button {
                background: linear-gradient(135deg, #f97316 0%, #ef4444 100%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 14px;
                font-weight: 700;
                color: #fff7ed;
                box-shadow: 0 12px 22px rgba(239, 68, 68, 0.25);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }
            .stButton > button:hover {
                box-shadow: 0 14px 30px rgba(249, 115, 22, 0.28);
                transform: translateY(-1px);
            }
            .stButton > button:focus {
                box-shadow: 0 0 0 3px rgba(125, 211, 252, 0.32);
            }
            .stTabs [role="tablist"] {
                gap: 0.6rem;
            }
            .stTabs [role="tab"] {
                border-radius: 10px 10px 0 0;
                padding: 0.55rem 0.85rem;
            }
            .stTabs [role="tab"][aria-selected="true"] {
                background: rgba(8, 145, 178, 0.12);
                color: #dff7ff;
                border-bottom: 2px solid rgba(34, 211, 238, 0.8);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("FLUX")
    st.caption("Lunar Image Registration & Localization")
    st.success("This app runs the real FLUX registration pipeline on the selected images.")

    source_upload, reference_upload = st.columns(2)
    with source_upload:
        with st.container(border=True):
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
        with st.container(border=True):
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