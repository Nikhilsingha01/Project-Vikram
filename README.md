# FLUX — Multi-Modal Lunar Image Registration

> **Structure-Guided, Terrain-Aware Image Correspondence for Chandrayaan-2 Optical Imagery**

FLUX is a software system designed to establish accurate correspondence between **Chandrayaan-2 optical images** and **lunar reference imagery** despite variations in illumination, scale, viewpoint, and image characteristics.

The system combines structure-guided image representation, terrain-aware candidate search, fine correspondence, evidence-based validation, geometric registration, and sub-pixel refinement to produce reliable and auditable image registration results.

---

## 🚀 Project Overview

Lunar images captured by different sensors or missions can differ significantly in:

- Sun illumination angle
- Viewing geometry
- Spatial scale
- Resolution
- Sensor characteristics
- Local terrain appearance

These variations make conventional image matching challenging.

FLUX addresses this problem through a multi-stage correspondence pipeline that first identifies structurally similar regions and then performs fine-grained matching and geometric validation.

### Core Pipeline

```text
Chandrayaan-2 Image
        │
        ▼
 Image Preparation
        │
        ▼
Structure-Guided Representation
        │
        ▼
Terrain-Aware Candidate Search
        │
        ▼
Fine Correspondence
        │
        ▼
Evidence-Gated Validation
        │
        ├───────────────┐
        │               │
     ACCEPT           REJECT
        │               │
        ▼               ▼
Geometric          Explainable
Registration       Match Failure
        │
        ▼
Sub-Pixel Refinement
        │
        ▼
Auditable Correspondence
        │
        ▼
Registered Image + Metrics

🎯 Problem Statement
Smart India Hackathon 2026

Problem Statement ID: SIH26166

Title:

Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

The objective is to develop a generic software solution capable of finding correspondence between Chandrayaan-2 optical imagery and lunar reference imagery while handling variations in:

Illumination
Scale
Viewing geometry
Sensor characteristics

The system should produce reliable correspondence points and a geometrically registered output with measurable accuracy.

💡 FLUX Approach

FLUX follows a structure-first, coarse-to-fine correspondence strategy rather than relying only on raw pixel similarity.

The system is organized into the following major stages:

Image Preparation
Structure-Guided Representation
Terrain-Aware Candidate Search
Fine Correspondence
Evidence-Gated Validation
Geometric Registration
Sub-Pixel Refinement
Auditable Correspondence
Explainable Match Failure
1. Image Preparation

Source and reference images are prepared to reduce differences caused by illumination, contrast, scale, and geometric characteristics.

Typical operations include:

Grayscale conversion
Intensity normalization
Contrast enhancement
CLAHE
Image scaling
Image tiling where required
Metadata preparation

The goal is to create a consistent representation suitable for subsequent structural analysis and correspondence.

2. Structure-Guided Representation

Instead of relying only on pixel intensity, FLUX extracts structural information from the lunar terrain.

The representation can incorporate:

Gradient information
Edge information
Orientation patterns
Local terrain structure
Multi-scale structural features

This provides a representation that can remain useful when illumination and intensity characteristics vary between images.

3. Terrain-Aware Candidate Search

Performing detailed correspondence across an entire lunar image can be computationally expensive.

FLUX therefore performs a coarse-to-fine search.

The candidate search stage:

Analyzes structural similarity
Performs coarse localization
Compares terrain structure
Filters unsuitable regions
Selects promising regions of interest

Only promising candidate regions are passed to the fine correspondence stage.

4. Fine Correspondence

Fine correspondence is performed on selected candidate regions.

The architecture supports experimentation with both classical and learned correspondence approaches.

Classical Baseline
SIFT
BFMatcher
Ratio test
Geometric filtering
Experimental Deep Learning Methods
SuperPoint
LoFTR
LightGlue

The initial MVP uses classical computer vision methods as a reliable baseline before introducing learned correspondence models.

5. Evidence-Gated Validation

A feature matcher alone does not guarantee a reliable correspondence.

FLUX therefore evaluates multiple forms of evidence before accepting a match.

Validation can consider:

Structural consistency
Geometric consistency
Scale consistency
Match confidence
Reprojection error
Inlier analysis
Spatial distribution of matches

The system then determines whether the correspondence should be accepted or rejected.

Candidate Correspondence
          │
          ▼
 ┌─────────────────────┐
 │ Evidence Evaluation │
 └──────────┬──────────┘
            │
      ┌─────┴─────┐
      │           │
   ACCEPT       REJECT
      │           │
      ▼           ▼
Registration   Failure
              Explanation
6. Explainable Match Failure

A failed registration should not simply return an error.

FLUX provides an interpretable reason for rejection whenever possible.

Possible failure reasons include:

Structural mismatch
Geometric mismatch
Scale inconsistency
Low match confidence
High reprojection error
Insufficient valid inliers

This makes the correspondence process more transparent and easier to debug.

7. Geometric Registration

Accepted correspondence points are used to estimate the geometric relationship between the source and reference images.

Depending on the image pair, the system can experiment with:

Affine transformation
Homography
RANSAC-based robust estimation

The estimated transformation is then applied to generate the registered image.

8. Sub-Pixel Refinement

After obtaining a stable geometric registration, accepted correspondence points can be refined at sub-pixel precision.

The objective is to improve the localization of corresponding points and achieve more accurate final alignment.

Sub-pixel refinement is performed after robust correspondence and geometric validation rather than being used as the initial matching mechanism.

9. Auditable Correspondence

FLUX records information about the correspondence process so that the final result can be inspected and evaluated.

A correspondence record can include:

Source Point
Reference Point
Match Confidence
Structural Evidence
Geometric Evidence
Validation Status
Transformation Information
Reprojection Error
Inlier Status
Acceptance / Rejection Information

This creates an auditable chain from the original images to the final registered output.

🛰️ Supported Imagery
Chandrayaan-2 Sources

FLUX is designed to support Chandrayaan-2 optical imagery including:

Sensor	Description
OHRC	Orbiter High Resolution Camera
TMC-2	Terrain Mapping Camera
IIRS	Imaging Infrared Spectrometer

The architecture keeps the sensor ingestion layer modular so that different Chandrayaan-2 datasets can be processed using the same overall pipeline.

Lunar Reference Imagery

The reference-image architecture supports lunar imagery from sources such as:

LRO
SELENE

Additional compatible lunar reference datasets can be integrated through the modular ingestion layer.

🏗️ System Architecture
                    ┌─────────────────────────┐
                    │   Chandrayaan-2 Data    │
                    │ OHRC / TMC-2 / IIRS     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Image Preparation    │
                    │ Normalize / Enhance     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Structure Representation│
                    │ Gradient / Edge / Shape │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Terrain-Aware Candidate │
                    │       Search            │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Fine Correspondence    │
                    │ SIFT / Deep Matchers    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Evidence-Gated          │
                    │ Validation              │
                    └────────────┬────────────┘
                                 │
                          ┌──────┴──────┐
                          │             │
                       ACCEPT        REJECT
                          │             │
                          ▼             ▼
                 ┌──────────────┐  ┌──────────────┐
                 │ Registration │  │   Failure    │
                 │ + Refinement │  │  Explanation │
                 └──────┬───────┘  └──────────────┘
                        │
                        ▼
                 ┌────────────────────┐
                 │    Final Output    │
                 │                    │
                 │ Registered Image   │
                 │ Match Points       │
                 │ RMSE               │
                 │ Inlier Count       │
                 │ Inlier Ratio       │
                 │ Spatial Coverage   │
                 └────────────────────┘
🧰 Technology Stack
Programming
Python
NumPy
SciPy
Computer Vision & Image Processing
OpenCV
Scikit-image
SIFT
RANSAC
Deep Learning / Experimental Matchers
PyTorch
SuperPoint
LoFTR
LightGlue
Geospatial & Satellite Data
GDAL
Rasterio
Visualization & Interface
Streamlit
Matplotlib
OpenCV visualization utilities
Hardware
Multi-core CPU
16 GB RAM recommended
NVIDIA GPU / CUDA for optional deep-learning experiments

Deep-learning components are optional experimental components and are not required for the initial classical MVP.

📁 Project Structure
FLUX/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .env.example
│
├── config/
│   ├── config.yaml
│   ├── sensors.yaml
│   └── thresholds.yaml
│
├── data/
│   ├── raw/
│   │   ├── chandrayaan2/
│   │   │   ├── OHRC/
│   │   │   ├── TMC-2/
│   │   │   └── IIRS/
│   │   │
│   │   └── lunar_reference/
│   │       ├── LRO/
│   │       └── SELENE/
│   │
│   ├── prepared/
│   ├── candidates/
│   ├── correspondences/
│   ├── registrations/
│   └── outputs/
│
├── src/
│   │
│   ├── main.py
│   ├── pipeline.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── source_loader.py
│   │   ├── reference_loader.py
│   │   └── metadata.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── normalization.py
│   │   ├── enhancement.py
│   │   └── tiling.py
│   │
│   ├── structure/
│   │   ├── __init__.py
│   │   ├── structure_representation.py
│   │   ├── edges.py
│   │   ├── gradients.py
│   │   └── terrain_features.py
│   │
│   ├── candidate_search/
│   │   ├── __init__.py
│   │   ├── coarse_localization.py
│   │   ├── terrain_filter.py
│   │   └── candidate_generator.py
│   │
│   ├── correspondence/
│   │   ├── __init__.py
│   │   ├── feature_matching.py
│   │   ├── superpoint.py
│   │   ├── loftr.py
│   │   └── lightglue.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── evidence_gating.py
│   │   ├── geometric_consistency.py
│   │   ├── inlier_analysis.py
│   │   └── rejection_reason.py
│   │
│   ├── registration/
│   │   ├── __init__.py
│   │   ├── geometric_registration.py
│   │   ├── ransac.py
│   │   └── subpixel_refinement.py
│   │
│   ├── audit/
│   │   ├── __init__.py
│   │   ├── correspondence_record.py
│   │   ├── metrics.py
│   │   ├── provenance.py
│   │   └── audit_report.py
│   │
│   ├── failure_analysis/
│   │   ├── __init__.py
│   │   ├── failure_classifier.py
│   │   ├── explain.py
│   │   └── failure_report.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── image_schema.py
│   │   ├── match_schema.py
│   │   ├── validation_schema.py
│   │   └── result_schema.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py
│       ├── geo_utils.py
│       ├── visualization.py
│       └── logging_utils.py
│
├── models/
│   ├── superpoint/
│   ├── loftr/
│   └── lightglue/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_structure_analysis.ipynb
│   ├── 04_candidate_search.ipynb
│   ├── 05_correspondence.ipynb
│   ├── 06_validation.ipynb
│   └── 07_registration_evaluation.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_structure.py
│   ├── test_candidate_search.py
│   ├── test_correspondence.py
│   ├── test_validation.py
│   ├── test_registration.py
│   └── test_audit.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── run_pipeline.py
│   └── evaluate_results.py
│
├── evaluation/
│   ├── ground_truth/
│   └── benchmark.py
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── generated/
│
└── app/
    ├── streamlit_app.py
    ├── components/
    └── pages/
📊 Evaluation Metrics

FLUX evaluates both correspondence quality and geometric registration quality.

RMSE

Root Mean Square Error measures the geometric error between corresponding points and the estimated transformation.

Lower RMSE generally indicates better geometric consistency.

Inlier Count

The number of correspondences that remain consistent with the estimated geometric model after robust estimation.

Inlier Ratio

The ratio of geometrically consistent correspondences to the total number of candidate correspondences.

Inlier Ratio =
Number of Inliers / Total Correspondences
Reprojection Error

Measures how closely transformed source points agree with their corresponding reference points.

Spatial Coverage

Measures the distribution of correspondence points across the image region.

A good correspondence set should not be concentrated only in a small local area.

🔍 Validation Strategy

FLUX uses multiple checks before accepting correspondence.

                 Candidate Matches
                        │
                        ▼
             ┌────────────────────┐
             │ Structural Check   │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Geometric Check    │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Scale Consistency  │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Confidence Check   │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Reprojection Error │
             └─────────┬──────────┘
                       │
                ┌──────┴──────┐
                │             │
             ACCEPT         REJECT
                │             │
                ▼             ▼
          Registration     Explainable
                          Match Failure
🧾 Audit Output

A successful registration can generate an audit record containing:

{
    source_image,
    reference_image,
    sensor,
    candidate_region,
    correspondence_points,
    confidence,
    transformation,
    inlier_count,
    inlier_ratio,
    rmse,
    reprojection_error,
    spatial_coverage,
    validation_status
}

This allows the result to be reproduced, inspected, and evaluated.

🖥️ FLUX Interface

The Streamlit application is designed as an interactive scientific image-registration interface.

Main Features
Source image upload
Reference image upload
Sensor selection
Image preview
Image preparation visualization
Structure analysis visualization
Candidate region visualization
Correspondence visualization
Validation status
Registered image visualization
Registration metrics
Match failure explanation
Audit information
Processing history
🔄 Main Application Workflow
┌───────────────┐
│ Upload Images │
└───────┬───────┘
        ↓
┌───────────────────┐
│ Image Preparation │
└────────┬──────────┘
         ↓
┌──────────────────────┐
│ Structure Analysis   │
└──────────┬───────────┘
           ↓
┌────────────────────────────┐
│ Terrain Candidate Search   │
└────────────┬───────────────┘
             ↓
┌─────────────────────┐
│ Fine Correspondence │
└──────────┬──────────┘
           ↓
┌──────────────────────────┐
│ Evidence-Gated Validation│
└────────────┬─────────────┘
             ↓
        ┌────┴────┐
        │         │
     ACCEPT     REJECT
        │         │
        ↓         ↓
 Registration   Explain
        │       Failure
        ↓
Sub-Pixel Refinement
        │
        ↓
Final Registered Image
        │
        ↓
Metrics + Audit Report
👥 Team Development Model

FLUX is designed for parallel development using well-defined module interfaces.

Member 1 — Data & Preprocessing

Responsibilities:

Dataset organization
Image ingestion
Metadata handling
Normalization
Enhancement
Image preparation
Structural maps
Member 2 — Candidate Search & Correspondence

Responsibilities:

Structure-based candidate search
Coarse localization
Terrain filtering
SIFT baseline
SuperPoint experiments
LoFTR experiments
LightGlue experiments
Match visualization
Member 3 — Validation & Registration

Responsibilities:

Evidence-gated validation
Geometric consistency
RANSAC
Transformation estimation
Registration
Sub-pixel refinement
RMSE
Inlier analysis
Spatial coverage
Member 4 — Frontend & Integration

Responsibilities:

Streamlit application
Image upload
Processing interface
Visualization
Result dashboard
Pipeline integration
Audit display
Failure explanation interface
🔗 Module Interfaces

The modules communicate through structured data contracts.

Preprocessing Output
{
    "image": image,
    "metadata": metadata,
    "sensor": sensor_type
}
Correspondence Output
{
    "source_points": source_points,
    "reference_points": reference_points,
    "confidence": confidence
}
Validation Output
{
    "accepted": True,
    "confidence": confidence,
    "reprojection_error": error,
    "rejection_reason": None
}
Registration Output
{
    "transformation": transformation,
    "registered_image": registered_image,
    "inlier_count": inlier_count,
    "inlier_ratio": inlier_ratio,
    "rmse": rmse,
    "spatial_coverage": spatial_coverage
}

These interfaces allow individual modules to be developed and tested independently.

🧪 Development Strategy

The project follows an incremental development strategy.

Phase 1 — Working Registration MVP
Image Input
     ↓
Preprocessing
     ↓
SIFT Matching
     ↓
RANSAC
     ↓
Affine / Homography
     ↓
Registered Image
Goal

Create a working end-to-end registration pipeline using a classical computer vision baseline.

Phase 2 — Structure & Candidate Search

Add:

Gradient representation
Edge representation
Orientation information
Multi-scale structural analysis
Coarse localization
Terrain-aware candidate filtering
Phase 3 — Evidence & Explainability

Add:

Evidence-gated validation
Match confidence
Geometric consistency
Reprojection error
Inlier analysis
Spatial coverage
Explainable rejection reasons
Audit records
Phase 4 — Sub-Pixel Refinement

Improve accepted correspondence points through local sub-pixel refinement.

Phase 5 — Deep Correspondence Experiments

Evaluate learned correspondence methods such as:

SuperPoint + LightGlue

and:

LoFTR

against the classical baseline.

📌 MVP Definition

The minimum successful FLUX prototype should be able to:

Accept a Chandrayaan-2 source image.
Accept a lunar reference image.
Prepare both images.
Generate structural representations.
Identify candidate regions.
Establish correspondence points.
Validate correspondence.
Estimate geometric transformation.
Produce a registered image.
Visualize match points.
Calculate RMSE.
Calculate inlier count.
Calculate inlier ratio.
Evaluate spatial coverage.
Explain why correspondence was rejected when validation fails.
🛡️ Design Principles
Structure First

Use terrain structure in addition to raw image intensity.

Coarse to Fine

Reduce the search space before performing expensive fine correspondence.

Evidence Before Acceptance

A correspondence should satisfy multiple consistency checks before being accepted.

Explainable Failure

Rejected correspondence should provide a meaningful explanation whenever possible.

Quantifiable Results

Registration quality should be measured using objective metrics.

Modular Architecture

Each pipeline stage should be independently testable and replaceable.

Reproducibility

Processing configuration, correspondence information, transformation parameters, and evaluation metrics should be recorded.

🔬 Research & Experimental Components

Some components of FLUX are intended for experimentation and benchmarking.

These include:

SuperPoint
LoFTR
LightGlue
Alternative feature detectors
Alternative geometric estimators
Different validation thresholds
Sub-pixel refinement strategies

The system maintains a classical computer vision baseline so that experimental methods can be evaluated against a known reference implementation.

📈 Future Improvements

Potential future development includes:

Advanced learned feature matching
Improved terrain-aware candidate search
Robust illumination normalization
Improved sub-pixel correspondence
Larger-scale benchmarking
Additional Chandrayaan-2 datasets
Automated parameter optimization
GPU acceleration
Improved audit and visualization capabilities
⚠️ Current Development Status

Prototype / Research Development

FLUX is being developed incrementally as a research-oriented prototype.

The initial implementation focuses on establishing a reliable classical computer vision baseline and a complete end-to-end registration pipeline.

Deep-learning correspondence methods and advanced refinement techniques are treated as experimental components.

Performance values will be reported only after evaluation on appropriate datasets and benchmark conditions.

🚀 Getting Started
1. Clone the Repository
git clone https://github.com/Nikhilsingha01/Project-Vikram.git
cd Project-Vikram
2. Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure the Project

Copy the example environment file:

cp .env.example .env

Configure the required parameters in:

config/config.yaml
config/sensors.yaml
config/thresholds.yaml
5. Prepare Data

Place source and reference imagery in the appropriate directories:

data/raw/chandrayaan2/
data/raw/lunar_reference/

Then run:

python scripts/prepare_dataset.py
6. Run the Pipeline
python scripts/run_pipeline.py
7. Launch the Streamlit Application
streamlit run app/streamlit_app.py

The application will provide the interactive FLUX registration interface.

## Run the MVP

From the project root, create and activate the virtual environment if needed,
then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch the complete Streamlit MVP with:

```bash
./run.sh
```

Open the URL shown by Streamlit. The app includes the repository's prototype
images, so the pipeline can be run without a browser upload.

🧪 Running Tests

Run the complete test suite using:

pytest

Run individual modules:

pytest tests/test_preprocessing.py
pytest tests/test_structure.py
pytest tests/test_candidate_search.py
pytest tests/test_correspondence.py
pytest tests/test_validation.py
pytest tests/test_registration.py
pytest tests/test_audit.py
📚 Notebooks

The repository contains notebooks for experimentation and analysis.

Notebook	Purpose
01_data_exploration.ipynb	Dataset exploration
02_preprocessing.ipynb	Image preparation
03_structure_analysis.ipynb	Structural representation
04_candidate_search.ipynb	Candidate localization
05_correspondence.ipynb	Feature matching
06_validation.ipynb	Correspondence validation
07_registration_evaluation.ipynb	Registration and evaluation
📂 Data Organization

Raw datasets should be kept separate from generated processing results.

data/
│
├── raw/
│   ├── chandrayaan2/
│   └── lunar_reference/
│
├── prepared/
├── candidates/
├── correspondences/
├── registrations/
└── outputs/

Generated files should not overwrite the original raw imagery.

Large datasets and restricted mission data should be handled according to their respective data-access and usage requirements.

🔐 Reproducibility & Auditability

For each processing run, FLUX aims to preserve:

Input image identifiers
Sensor information
Processing configuration
Candidate regions
Correspondence points
Validation evidence
Transformation parameters
Registration metrics
Acceptance / rejection status

This enables systematic evaluation and comparison between different configurations.

🤝 Contributing

Contributions are welcome.

A typical development workflow is:

Create Branch
     ↓
Implement Feature
     ↓
Add / Update Tests
     ↓
Run Test Suite
     ↓
Commit Changes
     ↓
Open Pull Request

Please keep changes modular and document new algorithms or processing stages.

📜 License

This project is released under the license specified in the LICENSE file.

🏆 Smart India Hackathon 2026
Project

FLUX

Problem Statement

SIH26166

Theme

Space Technology

Organization

Indian Space Research Organisation (ISRO)

Objective

Develop a robust software solution for correspondence and registration between Chandrayaan-2 optical imagery and lunar reference imagery under variations in illumination, scale, and viewing geometry.

👨‍🚀 Project Identity
███████╗██╗     ██╗   ██╗██╗  ██╗
██╔════╝██║     ██║   ██║╚██╗██╔╝
█████╗  ██║     ██║   ██║ ╚███╔╝
██╔══╝  ██║     ██║   ██║ ██╔██╗
██║     ███████╗╚██████╔╝██╔╝ ██╗
╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝

FLUX — Lunar Image Correspondence & Registration

Structure. Terrain. Correspondence. Evidence. Registration.

⭐ Key Concept
STRUCTURE
     +
TERRAIN
     +
CORRESPONDENCE
     +
EVIDENCE
     +
REGISTRATION
     =
FLUX

Accurate correspondence. Explainable validation. Reliable lunar image registration.

📌 Disclaimer

FLUX is a research and prototype software project developed for Smart India Hackathon 2026. The system's performance and accuracy depend on the input datasets, image characteristics, processing configuration, and selected correspondence/registration methods. Experimental methods and reported metrics should be validated on appropriate datasets before drawing operational conclusions.
