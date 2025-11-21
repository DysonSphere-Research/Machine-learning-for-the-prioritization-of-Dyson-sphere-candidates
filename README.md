# 🔭 Machine Learning Approach to Dyson Sphere Candidate Detection

*(Anonymous repository for double-blind peer review)*

This repository contains the complete codebase, documentation, and 
experimental framework used to develop a machine-learning approach for 
prioritizing **Dyson sphere candidates** using Gaia DR3 and WISE/AllWISE
photometry.

The project implements a **two-phase symmetric framework**, combining: -
a **Dyson-centric** learning phase, - a **Normal-centric** learning
phase, - and a **weighted fusion** scheme that merges the two into a
unified probabilistic ranking of stars.

Two independent pipelines are included:

-   **Isolation Forest pipeline** (`isolation_forest_pipeline.py`)
-   **Prototypical Network pipeline**
    (`PrototypicalNetwork_pipeline.py`)

Both are designed to be fully reproducible and follow the same data
format, evaluation logic, and ranking philosophy described in the
manuscript.

------------------------------------------------------------------------

# 📌 Manuscript Status

This repository accompanies a manuscript **currently prepared for
submission** to\
**Expert Systems with Applications (Elsevier)**.\
The repository is anonymized to comply with **double-blind peer-review**
guidelines.

No author names or affiliations are included.

------------------------------------------------------------------------

# 📁 Repository Structure

    📦 Machine-learning-approach-to-Dyson-sphere-detection
     ├── isolation_forest_pipeline.py
     ├── PrototypicalNetwork_pipeline.py
     ├── README.md                     # (this file)
     ├── README_PrototypicalNetwork.md # detailed PN documentation
     ├── reade.me                      # legacy instructions
     ├── data/                         # (user-provided, not included)
     └── results/                      # generated outputs

Each pipeline has its own dedicated documentation in this repository.

------------------------------------------------------------------------

# 🔧 Installation

## Requirements

-   Python ≥ 3.9\
-   numpy\
-   pandas\
-   scikit-learn\
-   torch (for Prototypical Networks)\
-   joblib

## Environment Setup

``` bash
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
.venv\Scripts\Activate.ps1        # Windows PowerShell

pip install -r requirements.txt
```

------------------------------------------------------------------------

# 📂 Input Data Format

The `data/` directory must contain:

    data/
      ├─ train.csv          # Dyson-centric support set
      ├─ trainNormal.csv    # Normal-centric support set
      ├─ test_normal.csv    # Evaluation set
      ├─ num_ds.txt         # number of Dyson spies at top of test_normal.csv
      ├─ numNorm.txt        # number of Normal spies immediately after Dyson block

### Requirements:

-   All files MUST include a **source_id** column\
-   Remaining columns must be **numeric features**\
-   Feature schemas MUST match across all files\
-   No missing values (NaN/Inf)

### Spy Protocol

-   First `num_ds` rows of `test_normal.csv` → **Dyson spies**\
-   Next `numNorm` rows → **Normal spies**

These are used exclusively for internal validation.

------------------------------------------------------------------------

# 🚀 Pipelines Overview

Both pipelines:

1.  Train a **Dyson-centric model**\
2.  Train a **Normal-centric model**\
3.  Generate two independent rankings\
4.  Fuse them using:

\[ `\text{score}`{=tex} =
`\alpha `{=tex}`\cdot `{=tex}`\text{DysonScore}`{=tex} +
(1-`\alpha`{=tex})`\cdot `{=tex}(1 - `\text{NormalScore}`{=tex}) \]

5.  Optionally compute **Precision/Recall/F1@k** metrics

------------------------------------------------------------------------

# 🌲 Isolation Forest Pipeline

File: `isolation_forest_pipeline.py`

### Example

``` bash
python isolation_forest_pipeline.py \
  --data-dir ./data \
  --out-dir ./results \
  --emit fused \
  --fuse-weights 0.9,0.7,0.5 \
  --emit-metrics --metrics-target dyson
```

------------------------------------------------------------------------

# 🧠 Prototypical Network Pipeline

File: `PrototypicalNetwork_pipeline.py`

### Example

``` bash
python PrototypicalNetwork_pipeline.py \
  --data-dir ./data \
  --out-dir ./results \
  --emit fused \
  --proto-method probability \
  --fuse-weights 0.9,0.7,0.5
```

------------------------------------------------------------------------

# 📤 Output Files

Generated in `results/`:

### Rankings

-   `*_ranking_dyson.csv`
-   `*_ranking_normal.csv`
-   `*_ranking_fused_wDyXX_wNXX_<norm>.csv`

### Metrics (optional)

-   `*_metrics_dyson.csv`
-   `*_metrics_normal.csv`

### Model Snapshots

-   `isoforest_model_*.joblib`
-   PN config + logs

### Execution Logs

-   `isoforest_*.log`
-   `protonet_*.log`


------------------------------------------------------------------------

# 📜 License

**All rights reserved.**\
This code is provided **exclusively** for manuscript evaluation and
peer-review purposes.\
No permission is granted for reuse, distribution, or modification.

------------------------------------------------------------------------

# ✨ Citation

Citation details will be added **after peer-review and acceptance**.\
During review, please cite this repository anonymously.
