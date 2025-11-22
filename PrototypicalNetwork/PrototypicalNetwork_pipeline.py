#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototypical Network

Usage (examples):
  # Run both regimes and save only the fused ranking with default weight (0.5)
  python prototypical_network_pipeline.py --data-dir /path/data --out-dir /path/out

  # Save dyson + normal + fused with multiple weights and metrics for Dyson as target
  python prototypical_network_pipeline.py --data-dir /path/data --out-dir /path/out \
      --emit dyson,normal,fused --fuse-weights 0.9,0.7,0.5,0.3 \
      --emit-metrics --metrics-target dyson

  # Save only dyson ranking using 'probability' scoring (softmax over proto distances)
  python prototypical_network_pipeline.py --data-dir /path/data --out-dir /path/out \
      --emit dyson --proto-method probability

Inputs in --data-dir:
  - train.csv           (Dyson-centered support set)     columns: source_id, <numeric features...>
  - trainNormal.csv     (Normal-centered support set)    columns: source_id, <numeric features...>
  - test_normal.csv     (evaluation set)                 columns: source_id, <numeric features...>
  - num_ds.txt          (first integer = number of Dyson spies at top of test)
  - numNorm.txt         (first integer = number of Normal spies after the Dyson block; for metrics target=normal)

Outputs in --out-dir (examples):
  - protonet_ranking_dyson.csv
  - protonet_ranking_normal.csv
  - protonet_ranking_fused_wDy0p70_wN0p30_minmax.csv
  - protonet_metrics_dyson.csv
  - protonet_metrics_normal.csv
  - protonet_dyson.log / protonet_normal.log / protonet_fused.log

Notes:
- CLI structure mirrors the Isolation Forest pipeline for consistency across methods.
- Add `--proto-method` to choose Prototypical scoring: 'probability', 'distance', 'cosine', or 'all'.
- Reproducible training via --seed; small MLP encoder to map features to an L2-normalized embedding.
"""

from __future__ import annotations
import argparse
import os
import re
import time
import json
from datetime import timedelta
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


# --------------------------- Logging ---------------------------

def _log(msg: str, log_file: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


# ---------------------- File helpers --------------------------

_INT_RE = re.compile(r"(-?\d+)")

def _read_int(path: str, key_hint: Optional[str] = None) -> int:
    """
    Read the first integer found in a text file (compatible with 'num_ds: 10' or plain '10').
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    txt = open(path, "r", encoding="utf-8").read()
    m = _INT_RE.search(txt)
    if not m:
        raise ValueError(f"No integer found in {path}. Content: {txt!r}")
    return int(m.group(1))


def _require_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")


# ---------------------- Data loading --------------------------

def load_data(
    data_dir: str,
    train_dyson: str = "train.csv",
    train_normal: str = "trainNormal.csv",
    test_name: str = "test_normal.csv",
    num_ds_file: str = "num_ds.txt",
    num_norm_file: str = "numNorm.txt",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """
    Load Dyson and Normal support sets, test set, and spy counts.
    The column 'source_id' must be present in all CSVs.
    """
    train_d = pd.read_csv(os.path.join(data_dir, train_dyson))
    train_n = pd.read_csv(os.path.join(data_dir, train_normal))
    test_df = pd.read_csv(os.path.join(data_dir, test_name))

    for df in (train_d, train_n, test_df):
        _require_columns(df, ["source_id"])

    num_ds = _read_int(os.path.join(data_dir, num_ds_file))
    num_norm = _read_int(os.path.join(data_dir, num_norm_file))

    return train_d, train_n, test_df, num_ds, num_norm


# -------------------- Spy labels --------------------------

def build_spy_labels(test_df: pd.DataFrame, num_ds: int, num_norm: int) -> pd.DataFrame:
    """
    Return a DataFrame with spy labels for each test star:
      - is_dyson_spy
      - is_normal_spy
    Collapses duplicate IDs with logical OR so labels are unique per ID.
    """
    n = len(test_df)
    is_dyson = np.zeros(n, dtype=bool)
    is_normal = np.zeros(n, dtype=bool)

    if num_ds > 0:
        is_dyson[:num_ds] = True
    if num_norm > 0:
        start, stop = num_ds, min(n, num_ds + num_norm)
        is_normal[start:stop] = True

    labels = pd.DataFrame({
        "ID": test_df["source_id"].values,
        "is_dyson_spy": is_dyson,
        "is_normal_spy": is_normal,
    })
    labels = (labels.groupby("ID", as_index=False)
                    .agg(is_dyson_spy=("is_dyson_spy", "any"),
                         is_normal_spy=("is_normal_spy", "any")))
    return labels


# -------------------- PN model -------------------------

class ProtoMLP(nn.Module):
    """
    Small MLP encoder that outputs L2-normalized embeddings.
    """
    def __init__(self, in_dim: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def prototypical_loss(support: torch.Tensor, support_y: torch.Tensor,
                      query: torch.Tensor, query_y: torch.Tensor) -> torch.Tensor:
    """
    Standard prototypical loss with softmax over negative squared distances.
    """
    classes = torch.unique(support_y)
    protos = torch.stack([support[support_y == c].mean(0) for c in classes], dim=0)  # [C, D]
    dists = torch.cdist(query, protos, p=2.0) ** 2
    log_p = F.log_softmax(-dists, dim=1)
    y_idx = torch.stack([(classes == yy).nonzero(as_tuple=False)[0] for yy in query_y]).squeeze(1)
    return F.nll_loss(log_p, y_idx)


def train_protonet(
    X_d: np.ndarray, X_n: np.ndarray,
    epochs: int, batch_size: int, lr: float,
    emb_dim: int, hidden: int, device: str,
    log_file: str
) -> Tuple[ProtoMLP, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Train a Prototypical Network encoder on Dyson/Normal support sets and return
    the trained model, both prototypes, and the fitted StandardScaler.
    """
    # Fit scaler on both supports
    scaler = StandardScaler().fit(np.vstack([X_d, X_n]))
    Xd = torch.from_numpy(scaler.transform(X_d)).float()
    Xn = torch.from_numpy(scaler.transform(X_n)).float()

    model = ProtoMLP(in_dim=Xd.shape[1], emb_dim=emb_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    y_d = torch.zeros(len(Xd), dtype=torch.long)
    y_n = torch.ones(len(Xn), dtype=torch.long)

    for ep in range(1, epochs + 1):
        model.train()
        ns = min(batch_size // 2, len(Xd))
        nt = min(batch_size // 2, len(Xn))
        idx_d = torch.randperm(len(Xd))[:ns]
        idx_n = torch.randperm(len(Xn))[:nt]
        sup_d = idx_d[: max(1, ns // 2)]
        sup_n = idx_n[: max(1, nt // 2)]
        que_d = idx_d[max(1, ns // 2):]
        que_n = idx_n[max(1, nt // 2):]

        support = torch.cat([Xd[sup_d], Xn[sup_n]], dim=0).to(device)
        support_y = torch.cat([y_d[sup_d], y_n[sup_n]], dim=0).to(device)
        query = torch.cat([Xd[que_d], Xn[que_n]], dim=0).to(device)
        query_y = torch.cat([y_d[que_d], y_n[que_n]], dim=0).to(device)

        loss = prototypical_loss(model(support), support_y, model(query), query_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if ep == 1 or ep % max(1, epochs // 10) == 0:
            _log(f"Epoch {ep:03d}/{epochs} loss={loss.item():.4f}", log_file)

    model.eval()
    with torch.no_grad():
        proto_d = model(Xd.to(device)).mean(0, keepdim=True)  # [1, D]
        proto_n = model(Xn.to(device)).mean(0, keepdim=True)

    return model, proto_d, proto_n, scaler


@torch.no_grad()
def score_streams(
    z_te: torch.Tensor, proto_d: torch.Tensor, proto_n: torch.Tensor, method: str
) -> Dict[str, np.ndarray]:
    """
    Compute Dyson- and Normal-likeness streams according to the chosen method.

    Methods:
      - 'probability': 2-way softmax over negative squared distances -> P(Dyson), P(Normal)
      - 'distance'   : use -||z - proto||^2 (higher is better = closer)
      - 'cosine'     : cosine similarity to prototype in [-1,1] rescaled to [0,1]
    Returns dict with keys: 'dyson', 'normal'
    """
    assert method in {"probability", "distance", "cosine"}

    if method == "probability":
        d_d = (torch.cdist(z_te, proto_d, p=2.0) ** 2).squeeze(1)
        d_n = (torch.cdist(z_te, proto_n, p=2.0) ** 2).squeeze(1)
        logits = torch.stack([-d_d, -d_n], dim=1)
        prob = F.softmax(logits, dim=1).cpu().numpy()
        return {"dyson": prob[:, 0], "normal": prob[:, 1]}

    if method == "distance":
        s_d = -(torch.cdist(z_te, proto_d, p=2.0) ** 2).squeeze(1).cpu().numpy()
        s_n = -(torch.cdist(z_te, proto_n, p=2.0) ** 2).squeeze(1).cpu().numpy()
        return {"dyson": s_d, "normal": s_n}

    # cosine
    s_d = (z_te @ proto_d.T).squeeze(1).cpu().numpy()
    s_n = (z_te @ proto_n.T).squeeze(1).cpu().numpy()
    # map [-1,1] -> [0,1] for compatibility with fusion
    s_d = (s_d + 1.0) / 2.0
    s_n = (s_n + 1.0) / 2.0
    return {"dyson": s_d, "normal": s_n}


# -------------------- Ranking & Fusion -------------------------

def normalize_series(s: pd.Series, scheme: str) -> pd.Series:
    if scheme == "none":
        return s.astype(float)
    if scheme == "zscore":
        mu, sigma = s.mean(), s.std(ddof=0)
        return (s - mu) / (sigma if sigma > 0 else 1.0)
    # minmax
    smin, smax = float(s.min()), float(s.max())
    if smax == smin:
        return pd.Series(np.zeros_like(s, dtype=float), index=s.index)
    return (s - smin) / (smax - smin)


def format_weight(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def fuse_rankings(
    dyson_rank: pd.DataFrame,
    normal_rank: pd.DataFrame,
    alpha_dyson: float,
    norm_scheme: str = "minmax",
) -> pd.DataFrame:
    """
    Classic Dyson-vs-Normal fusion:
      fused = alpha * norm(dyson) + (1 - alpha) * (1 - norm(normal))
    """
    df = pd.merge(dyson_rank.rename(columns={"score": "score_dyson"}),
                  normal_rank.rename(columns={"score": "score_normal"}),
                  on="ID", how="inner", validate="one_to_one")

    s_d = normalize_series(df["score_dyson"].astype(float), norm_scheme)
    s_n = normalize_series(df["score_normal"].astype(float), norm_scheme)
    fused = alpha_dyson * s_d + (1.0 - alpha_dyson) * (1.0 - s_n)

    fused_df = pd.DataFrame({"ID": df["ID"], "score": fused})
    fused_df = fused_df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    return fused_df


def ensure_unique_ids(ranking: pd.DataFrame, policy: str, label: str) -> pd.DataFrame:
    """
    Ensure the 'ID' column is unique with the selected policy.
    """
    if ranking["ID"].is_unique:
        return ranking
    dup_cnt = ranking["ID"].duplicated().sum()
    print(f"[WARN] {label}: found {dup_cnt} duplicated IDs. Applying policy: {policy}")
    if policy == "error":
        raise ValueError(f"Duplicate IDs in {label} ranking.")
    if policy == "drop-keep-best":
        return ranking.sort_values("score", ascending=False).drop_duplicates("ID").reset_index(drop=True)
    if policy in {"mean", "max", "min"}:
        rk = ranking.groupby("ID", as_index=False).agg(score=("score", policy))
        return rk.sort_values("score", ascending=False).reset_index(drop=True)
    return ranking.drop_duplicates("ID").reset_index(drop=True)


def save_ranking(out_dir: str, name: str, df: pd.DataFrame) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    return path


# -------------------- Metrics (Precision/Recall/F1) ------------------------

def spy_mask(n_test: int, num_ds: int, num_norm: int, target: str) -> np.ndarray:
    mask = np.zeros(n_test, dtype=bool)
    if target == "dyson":
        mask[:max(0, num_ds)] = True
    else:
        start, stop = max(0, num_ds), min(n_test, num_ds + max(0, num_norm))
        mask[start:stop] = True
    return mask


def metrics_from_ranking(
    ranking: pd.DataFrame,
    test_df: pd.DataFrame,
    num_ds: int,
    num_norm: int,
    metrics_target: str,
) -> pd.DataFrame:
    """
    Compute cumulative Precision/Recall/F1 for each k (1..N) using the implicit spy mask
    over the original test order. Robust to duplicate source_id in test_df.
    """
    pos_mask = spy_mask(len(test_df), num_ds, num_norm, metrics_target)

    # Map: ID -> first position in test order (handles duplicate IDs)
    pos_map = (
        pd.Series(np.arange(len(test_df)), index=test_df["source_id"].values)
          .groupby(level=0)
          .first()
    )

    rnk = ranking.copy()
    rnk["pos"] = rnk["ID"].map(pos_map)
    rnk = rnk.dropna(subset=["pos"]).copy()
    rnk["pos"] = rnk["pos"].astype(int)

    is_tp_ordered = pos_mask[rnk["pos"].values]
    cum_tp = np.cumsum(is_tp_ordered.astype(int))
    k = np.arange(1, len(rnk) + 1)
    denom_pos = int(pos_mask.sum()) if int(pos_mask.sum()) > 0 else 1
    precision = cum_tp / k
    recall = cum_tp / denom_pos
    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)

    return pd.DataFrame({"k": k, "precision": precision, "recall": recall, "f1": f1})


# ------------------------------ CLI --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prototypical Network (dyson | normal) with weighted fusion")
    # Match the Isolation Forest CLI structure
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--emit", type=str, default="fused_only")  # {dyson,normal,fused}, comma-separated
    p.add_argument("--fuse-weights", type=str, default="0.50")
    p.add_argument("--norm-scheme", type=str, choices=["minmax", "zscore", "none"], default="minmax")
    p.add_argument("--emit-metrics", action="store_true")
    p.add_argument("--metrics-target", type=str, choices=["dyson", "normal"], default="dyson")
    p.add_argument("--train-file-dyson", type=str, default="train.csv")
    p.add_argument("--train-file-normal", type=str, default="trainNormal.csv")
    p.add_argument("--test-file", type=str, default="test_normal.csv")
    p.add_argument("--num-ds-file", type=str, default="num_ds.txt")
    p.add_argument("--num-norm-file", type=str, default="numNorm.txt")
    p.add_argument("--on-duplicate", type=str, choices=["error", "drop-keep-best", "mean", "max", "min"],
                   default="drop-keep-best")

    # PN-specific hyperparameters (added without altering the core CLI structure above)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

    # New: choose PN scoring method
    p.add_argument("--proto-method", type=str,
                   choices=["probability", "distance", "cosine", "all"],
                   default="probability")

    return p.parse_args()


# ------------------------------ Main --------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)

    emit = {e.strip().lower() for e in args.emit.split(",")}
    if emit == {"fused_only"}:
        emit = {"fused"}

    t0 = time.time()

    # Load data and spies
    train_d, train_n, test_df, num_ds, num_norm = load_data(
        args.data_dir, args.train_file_dyson, args.train_file_normal,
        args.test_file, args.num_ds_file, args.num_norm_file
    )
    # Features / IDs
    Xd_np = train_d.drop(columns=["source_id"]).to_numpy(dtype=float, copy=False)
    Xn_np = train_n.drop(columns=["source_id"]).to_numpy(dtype=float, copy=False)
    Xte_np = test_df.drop(columns=["source_id"]).to_numpy(dtype=float, copy=False)
    ids_te = test_df["source_id"].values

    spy_labels = build_spy_labels(test_df, num_ds, num_norm)
    logs = {m: os.path.join(args.out_dir, f"protonet_{m}.log") for m in ["dyson", "normal", "fused"]}
    for k in logs: open(logs[k], "w").write(f"Prototypical Network ({k}) — log\n")

    _log(f"Device: {device}", logs["dyson"])
    _log(f"Train Dyson shape={Xd_np.shape} | Train Normal shape={Xn_np.shape} | Test shape={Xte_np.shape}", logs["dyson"])

    # Train PN
    model, proto_d, proto_n, scaler = train_protonet(
        X_d=Xd_np, X_n=Xn_np,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        emb_dim=args.embedding_dim, hidden=args.hidden_dim,
        device=device, log_file=logs["dyson"]
    )

    # Embed test set
    model.eval()
    with torch.no_grad():
        Xte_scaled = torch.from_numpy(scaler.transform(Xte_np)).float().to(device)
        z_te = model(Xte_scaled)  # [N, D]

    # Which scoring methods to run
    methods = [args.proto_method] if args.proto_method != "all" else ["probability", "distance", "cosine"]

    for method in methods:
        _log(f"Scoring method = {method}", logs["dyson"])

        # Streams
        streams = score_streams(z_te, proto_d, proto_n, method=method)
        dyson_scores = streams["dyson"]
        normal_scores = streams["normal"]

        # Dyson ranking
        rank_d = (pd.DataFrame({"ID": ids_te, "score": dyson_scores})
                    .sort_values("score", ascending=False, kind="mergesort")
                    .reset_index(drop=True))
        rank_d = ensure_unique_ids(rank_d, args.on_duplicate, f"dyson[{method}]")

        # Normal ranking (higher = more normal-like)
        rank_n = (pd.DataFrame({"ID": ids_te, "score": normal_scores})
                    .sort_values("score", ascending=False, kind="mergesort")
                    .reset_index(drop=True))
        rank_n = ensure_unique_ids(rank_n, args.on_duplicate, f"normal[{method}]")

        # Save Dyson ranking
        if "dyson" in emit:
            out_d = rank_d.merge(spy_labels[["ID", "is_dyson_spy"]], on="ID", how="left")
            name_d = "protonet_ranking_dyson.csv" if args.proto_method != "all" else f"protonet_ranking_dyson_{method}.csv"
            save_ranking(args.out_dir, name_d, out_d)
            _log(f"Saved Dyson ranking: {name_d}", logs["dyson"])

        # Save Normal ranking
        if "normal" in emit:
            out_n = rank_n.merge(spy_labels[["ID", "is_dyson_spy", "is_normal_spy"]], on="ID", how="left")
            name_n = "protonet_ranking_normal.csv" if args.proto_method != "all" else f"protonet_ranking_normal_{method}.csv"
            save_ranking(args.out_dir, name_n, out_n)
            _log(f"Saved Normal ranking: {name_n}", logs["normal"])

        # Save Fused rankings for all alphas requested
        if "fused" in emit:
            alphas = [float(x) for x in args.fuse_weights.split(",")]
            for a in alphas:
                fused_df = fuse_rankings(rank_d, rank_n, a, args.norm_scheme)
                fused_df = fused_df.merge(spy_labels, on="ID", how="left")
                name_f = (
                    f"protonet_ranking_fused_wDy{format_weight(a)}_wN{format_weight(1.0-a)}_{args.norm_scheme}.csv"
                    if args.proto_method != "all"
                    else f"protonet_ranking_fused_{method}_wDy{format_weight(a)}_wN{format_weight(1.0-a)}_{args.norm_scheme}.csv"
                )
                save_ranking(args.out_dir, name_f, fused_df)
                _log(f"Saved fused ranking: {name_f}", logs["fused"])

                # Optional metrics
                if args.emit_metrics:
                    m = metrics_from_ranking(fused_df, test_df, num_ds, num_norm, args.metrics_target)
                    metrics_name = (
                        f"protonet_metrics_{args.metrics_target}_wDy{format_weight(a)}_{args.norm_scheme}.csv"
                        if args.proto_method != "all"
                        else f"protonet_metrics_{args.metrics_target}_{method}_wDy{format_weight(a)}_{args.norm_scheme}.csv"
                    )
                    save_ranking(args.out_dir, metrics_name, m)
                    _log(f"Saved metrics: {metrics_name}", logs["fused"])

    # Save run config snapshot
    with open(os.path.join(args.out_dir, "protonet_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dt = timedelta(seconds=int(time.time() - t0))
    for k in logs:
        _log(f"Elapsed: {dt}", logs[k])


if __name__ == "__main__":
    main()


