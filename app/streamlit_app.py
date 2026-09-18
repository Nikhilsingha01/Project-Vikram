"""Streamlit entry point for the FLUX demonstration application."""

from io import BytesIO
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
from app.pipeline import run_flux


SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]


def main() -> None:
    """Render the FLUX application."""
    st.set_page_config(
        page_title="FLUX | Lunar Image Registration",
        page_icon="🌙",
        layout="wide",
    )
    st.title("FLUX")
    st.caption("Lunar Image Registration & Localization")
    st.warning(
        "The current application runs in DEMO MODE. It validates the UI flow "
        "but does not perform scientific matching or registration."
    )

    source_upload, reference_upload = st.columns(2)
    with source_upload:
        source_file = st.file_uploader(
            "Source Image", type=SUPPORTED_TYPES, key="source_image"
        )
    with reference_upload:
        reference_file = st.file_uploader(
            "Reference Image", type=SUPPORTED_TYPES, key="reference_image"
        )

    source_image = _read_uploaded_image(source_file, "source")
    reference_image = _read_uploaded_image(reference_file, "reference")
    if source_image is not None and reference_image is not None:
        display_image_comparison(source_image, reference_image)

    if st.button("Run FLUX", type="primary", use_container_width=True):
        if source_image is None or reference_image is None:
            st.error("Upload both a source image and a reference image before running FLUX.")
            return
        try:
            result = run_flux(source_image, reference_image)
        except (TypeError, ValueError) as error:
            st.error(f"Unable to process the uploaded images: {error}")
            return
        _display_result(result)


def _read_uploaded_image(uploaded_file: Optional[Any], label: str) -> Optional[np.ndarray]:
    """Decode an uploaded image without exposing decoder tracebacks in the UI."""
    if uploaded_file is None:
        return None
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error(f"The uploaded {label} file is not a valid supported image.")
        return None
    return np.asarray(image)


def _display_result(result: Dict[str, Any]) -> None:
    """Render a standard pipeline result."""
    st.divider()
    st.header("Result")
    display_processing_status(result.get("processing_steps"))
    display_image_comparison(
        result.get("source_image"),
        result.get("reference_image"),
        result.get("registered_image"),
    )
    display_metrics(result)
    display_match_status(result)


if __name__ == "__main__":
    main()