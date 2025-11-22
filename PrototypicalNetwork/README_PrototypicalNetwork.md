# Prototypical Network for Dyson Sphere Candidate Discovery

This repository provides a publication-ready Python pipeline to train and evaluate
**Prototypical Networks** for Dyson Sphere candidate detection. It supports two
training regimes (Dyson-centric and Normal-centric) and a **weighted fusion** of
their rankings into a single, unified list. You can also choose the **scoring method**
used by the ProtoNet (`probability`, `distance`, `cosine`, or `all`).

 


## 🔧 Installation

**Requirements**
- Python ≥ 3.9
- `numpy`, `pandas`, `scikit-learn`, `torch` (CUDA optional)

**Install**
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Minimal `requirements.txt` example:
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
torch>=2.0
```


## 📂 Input data layout

Place your files in a single directory (e.g. `./data`):

```
data/
  ├─ train.csv          # Dyson-centric support set (must include 'source_id' + numeric features)
  ├─ trainNormal.csv    # Normal-centric support set (must include 'source_id' + numeric features)
  ├─ test_normal.csv    # Evaluation set (must include 'source_id' + numeric features)
  ├─ num_ds.txt         # integer: number of Dyson spies at the TOP of test_normal.csv
  ├─ numNorm.txt        # integer: number of Normal spies immediately AFTER the Dyson block
```

**CSV schema**
- Comma-separated, with header.
- First column: **`source_id`** (unique per row).
- Remaining columns: **numeric features** only (float/int).
- No missing values (NaN/Inf).
- Consistent feature names and order across all files.

**Spy convention (implicit labels used for evaluation)**
- Dyson spies = first `num_ds` rows of `test_normal.csv`.
- Normal spies = rows `[num_ds : num_ds + numNorm)` of `test_normal.csv`.


## 💻 Usage

Basic run (default = **fused ranking only**, Dyson weight 0.50, min-max normalization):
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results
```

Emit specific outputs:
```bash
# only Dyson ranking
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit dyson

# Dyson + Normal rankings
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit dyson,normal

# Dyson + Normal + fused ranking
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit dyson,normal,fused
```

Fused ranking with custom weights (Dyson weight α, Normal weight = 1−α):
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --emit fused --fuse-weights 0.9,0.7,0.5
```

Choose the ProtoNet scoring method:
```bash
# probability (softmax over negative squared distances to the two prototypes)
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit fused --proto-method probability

# distance (use -||z - prototype||^2, higher is better = closer)
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit fused --proto-method distance

# cosine (cosine similarity in [-1,1] rescaled to [0,1])
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit fused --proto-method cosine

# run all three and save outputs with method suffixes
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results --emit dyson,normal,fused --proto-method all
```

Metrics (Precision/Recall/F1@k) on the fused ranking:
```bash
python prototypical_network_pipeline.py --data-dir ./data --out-dir ./results \
  --emit fused --emit-metrics --metrics-target dyson
```

Windows (PowerShell) example with line wrapping:
```powershell
py .\prototypical_network_pipeline.py --data-dir .\data --out-dir .\results `
  --emit dyson,normal,fused --proto-method all --fuse-weights 0.9
```


## ⚙️ Command-line options

**Core (aligned with the Isolation Forest pipeline)**
- `--data-dir` path to the folder with input files (**required**)
- `--out-dir` path to the folder for outputs (**required**)
- `--emit` what to save:
  - `dyson` | `normal` | `fused` | `fused_only` | `dyson,normal` | `dyson,normal,fused`
  - default: `fused_only` (computes both regimes, saves only the fused ranking)
- `--fuse-weights` comma-separated Dyson weights (e.g., `0.9,0.7,0.5`); Normal weight = `1−α`
- `--norm-scheme` normalization for fusion: `minmax | zscore | none` (default `minmax`)
- `--emit-metrics` save Precision/Recall/F1@k CSV for fused ranking
- `--metrics-target` positive class for metrics: `dyson | normal` (default `dyson`)

**Custom file names (optional)**
- `--train-file-dyson` (default `train.csv`)
- `--train-file-normal` (default `trainNormal.csv`)
- `--test-file` (default `test_normal.csv`)
- `--num-ds-file` (default `num_ds.txt`)
- `--num-norm-file` (default `numNorm.txt`)

**Duplicates policy (before fusion)**
- `--on-duplicate` `{error | drop-keep-best | mean | max | min}` (default `drop-keep-best`)

**ProtoNet specifics**
- `--proto-method` `{probability | distance | cosine | all}` (default `probability`)
- `--epochs` (default 100)
- `--batch-size` (default 64)
- `--lr` (default 5e-4)
- `--embedding-dim` (default 64)
- `--hidden-dim` (default 128)
- `--seed` (default 42)
- `--device` `{auto | cpu | cuda}` (default `auto`)


## 📊 Outputs

All files are written to `--out-dir`. Examples:

- `protonet_ranking_dyson.csv`  
  Columns: `ID, score, is_dyson_spy`

- `protonet_ranking_normal.csv`  
  Columns: `ID, score, is_dyson_spy, is_normal_spy`

- `protonet_ranking_fused_wDy0p90_wN0p10_minmax.csv`  
  Columns: `ID, score, is_dyson_spy, is_normal_spy`  
  *(weights and normalization scheme are encoded in the filename; if `--proto-method all`, the method name is appended.)*

- (optional) `protonet_metrics_{dyson|normal}[_METHOD]_wDy{α}_{norm}.csv`  
  Columns: `k, precision, recall, f1` (cumulative @k)

- Logs and config snapshot:  
  `protonet_dyson.log`, `protonet_normal.log`, `protonet_fused.log`  
  `protonet_config.json`


## 🧪 Notes & design choices

- **Reproducibility**: full control via `--seed` and deterministic CUDA/NN flags.
- **Two regimes**: Dyson-centric and Normal-centric are learned via class prototypes; the encoder is a compact MLP with L2-normalized embeddings.
- **Scoring methods**:
  - `probability`: 2-way softmax over negative squared distances → P(Dyson) vs P(Normal)
  - `distance`: uses `-||z − prototype||²` (higher = closer)
  - `cosine`: cosine similarity rescaled to [0,1]
- **Fusion**: `fused = α·norm(Dyson) + (1−α)·(1−norm(Normal))`, with `norm ∈ {minmax, zscore, none}`.
- **Duplicates**: enforce unique `ID` before fusion with `--on-duplicate` policy.

 
