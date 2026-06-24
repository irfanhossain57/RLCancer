"""
src/predict.py
==============
Command-line prediction script.
Loads trained models, runs the full inference pipeline,
and prints the recommended drug treatment sequence.

CHANGES FROM ORIGINAL:
  - File path lookup updated: looks for no-suffix files first
    (gat_model.pt, cell_graph.npz, processed_adata.h5ad)
    which match Kaggle notebook output — falls back to _exp1
  - DRUG_NAMES updated: Venetoclax, Imatinib, Panobinostat,
    Dasatinib, Quizartinib
  - CancerDrugEnv called with DataFrame, not CSV path + cfg

Usage:
    python src/predict.py
    python src/predict.py --input models/processed_adata.h5ad
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import anndata as ad
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from src.train import GATv2Classifier, sparse_to_edge_index

parser = argparse.ArgumentParser(
    description="SC-RLOT: Predict drug resistance and optimal treatment sequence"
)
parser.add_argument(
    "--input", default=None,
    help="Path to a processed .h5ad file (default: auto-detect best model)"
)
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

DEVICE     = torch.device("cpu")
SEED       = cfg["subsample"]["random_seed"]

# CHANGE: updated drug list — matches Kaggle notebook and config.yaml
DRUG_NAMES = cfg["gdsc2"]["drugs"]   # Venetoclax, Imatinib, Panobinostat, Dasatinib, Quizartinib

np.random.seed(SEED)

print("\n" + "=" * 58)
print("  SC-RLOT: Cancer Drug Resistance Predictor")
print("=" * 58)


# ══════════════════════════════════════════════════════════════
# FILE DISCOVERY
# CHANGE: search order updated — no-suffix files (Kaggle) first
# ══════════════════════════════════════════════════════════════

def find_files():
    """
    Return (h5ad_path, gat_path, graph_path) for the best available model.
    Search order:
        1. models/processed_adata.h5ad   + models/gat_model.pt   (Kaggle / exp2 tuned)
        2. models/processed_adata_exp2.h5ad  (old suffix style)
        3. models/processed_adata_exp1.h5ad  (local baseline fallback)
    """
    candidates = [
        ("models/processed_adata.h5ad",
         "models/gat_model.pt",
         "models/cell_graph.npz"),
        ("models/processed_adata_exp2.h5ad",
         "models/gat_model_exp2.pt",
         "models/cell_graph_exp2.npz"),
        ("models/processed_adata_exp1.h5ad",
         "models/gat_model_exp1.pt",
         "models/cell_graph_exp1.npz"),
    ]

    # If user passed --input, honour it but still auto-find gat/graph
    if args.input:
        if not os.path.exists(args.input):
            print(f"\n[ERROR] Input not found: {args.input}")
            print("  Run 'python src/train.py' first.")
            sys.exit(1)
        # find matching gat + graph
        for h5ad, gat, graph in candidates:
            if os.path.exists(gat) and os.path.exists(graph):
                return args.input, gat, graph
        # last resort — return what we have even if graph is missing
        return args.input, candidates[0][1], candidates[0][2]

    for h5ad, gat, graph in candidates:
        if os.path.exists(h5ad) and os.path.exists(gat):
            return h5ad, gat, graph

    print("\n[ERROR] No trained model files found in models/")
    print("  Run 'python src/train.py --config configs/config.yaml' first.")
    sys.exit(1)


h5ad_path, gat_path, graph_path = find_files()


# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
print(f"\n[1] Loading: {h5ad_path}")
adata = ad.read_h5ad(h5ad_path)
print(f"  {adata.n_obs} cells × {adata.n_vars} genes")
print(f"  Subclones : {adata.obs['subclone'].nunique()}")
resist_col = "resistant" if "resistant" in adata.obs else None
if resist_col:
    print(f"  Resistant : {adata.obs['resistant'].sum()} "
          f"({adata.obs['resistant'].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# 2. GATv2 RESISTANCE CLASSIFICATION
# ══════════════════════════════════════════════════════════════
print(f"\n[2] Running GATv2 resistance classifier...")
print(f"    Model:  {gat_path}")
print(f"    Graph:  {graph_path}")

if not os.path.exists(graph_path):
    print(f"  [ERROR] Cell graph not found: {graph_path}")
    sys.exit(1)

adj        = sp.load_npz(graph_path)
edge_index = sparse_to_edge_index(adj).to(DEVICE)
X          = torch.tensor(adata.obsm["X_vae"], dtype=torch.float32).to(DEVICE)

model_gat = GATv2Classifier(
    in_dim=cfg["model"]["n_latent"],
    hidden_dim=cfg["model"]["gat_hidden_dim"],
    n_heads=cfg["model"]["gat_heads"],
).to(DEVICE)
model_gat.load_state_dict(torch.load(gat_path, map_location=DEVICE))
model_gat.eval()

with torch.no_grad():
    logits = model_gat(X, edge_index)
    probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

adata.obs["resist_prob"] = probs
n_resist = (probs > 0.5).sum()
print(f"  Resistant cells detected: {n_resist} ({n_resist/adata.n_obs*100:.1f}%)")

# Subclone summary
sc_summary = adata.obs.groupby("subclone").agg(
    n_cells=("resist_prob", "count"),
    mean_resist=("resist_prob", "mean"),
).reset_index()

print(f"\n  Subclone resistance summary:")
for _, row in sc_summary.iterrows():
    is_most = row["mean_resist"] == sc_summary["mean_resist"].max()
    tag = " ← MOST RESISTANT" if is_most else ""
    print(f"    Subclone {row['subclone']}: {int(row['n_cells']):4d} cells "
          f"| resist_prob={row['mean_resist']:.3f}{tag}")


# ══════════════════════════════════════════════════════════════
# 3. PPO DRUG RECOMMENDATION
# ══════════════════════════════════════════════════════════════
print(f"\n[3] Running PPO drug recommendation engine...")

from stable_baselines3 import PPO
from models.rl_env import CancerDrugEnv

# CHANGE: load GDSC merged CSV → pass as DataFrame (not path + cfg)
gdsc_csv = "data/gdsc_merged_ic50.csv"
gdsc_df  = pd.read_csv(gdsc_csv) if os.path.exists(gdsc_csv) else None
if gdsc_df is None:
    print("  [WARN] GDSC merged CSV not found — using random drug effects")
    print("         Run 'python src/train.py' to generate data/gdsc_merged_ic50.csv")

# CHANGE: CancerDrugEnv now takes (adata, gdsc_df) not (adata, csv_path, cfg)
env = CancerDrugEnv(adata, gdsc_df)

# CHANGE: PPO file lookup — no suffix first (Kaggle), then _exp1
ppo_candidates = [
    "models/ppo_model.zip",
    "models/ppo_model_exp2.zip",
    "models/ppo_model_exp1.zip",
]
ppo_path   = next((p for p in ppo_candidates if os.path.exists(p)), None)
use_random = False

if ppo_path:
    print(f"  PPO model: {ppo_path}")
    ppo_model = PPO.load(ppo_path, device=DEVICE)
else:
    print("  [WARN] No PPO model found — using random policy for demo")
    print("         Run 'python src/train.py' first for a trained agent")
    use_random = True

obs, _ = env.reset()
print(f"\n  Initial tumour state (subclone proportions):")
for i, (clone, prop) in enumerate(zip(env.clone_ids, obs)):
    resist = env.subclone_resist[i]
    flag   = " ← TARGET" if i == env.resistant_subclone else ""
    print(f"    Subclone {clone}: {prop:.3f}  (resist={resist:.3f}){flag}")

print(f"\n  Recommended treatment sequence:")
print(f"  {'Step':>4} | {'Drug':>14} | {'Resist Prop':>11} | {'Reward':>8}")
print(f"  {'-'*46}")

total_reward   = 0.0
initial_resist = float(obs[env.resistant_subclone])

for step in range(cfg["model"]["episode_length"]):
    if use_random:
        action = env.action_space.sample()
    else:
        action, _ = ppo_model.predict(obs, deterministic=True)
        action = int(action)

    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    print(f"  {step+1:>4} | {info['drug_used']:>14} | "
          f"{info['resist_prop']:>11.4f} | {reward:>8.4f}")

final_resist = float(info["resist_prop"])
reduction    = (initial_resist - final_resist) / (initial_resist + 1e-8) * 100

# Drug usage summary
drug_counts = {}
for h in env.episode_history:
    drug_counts[h["drug"]] = drug_counts.get(h["drug"], 0) + 1


# ══════════════════════════════════════════════════════════════
# RESULT SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*58}")
print(f"  PREDICTION RESULT")
print(f"{'='*58}")
print(f"  Total treatment reward    :  {total_reward:.4f}")
print(f"  Resistant subclone (start):  {initial_resist:.4f}")
print(f"  Resistant subclone (end)  :  {final_resist:.4f}")
print(f"  Resistance suppression    :  {reduction:.1f}%")
print(f"\n  Drug usage across {cfg['model']['episode_length']} treatment steps:")
for drug, count in sorted(drug_counts.items(), key=lambda x: -x[1]):
    bar = "█" * count
    print(f"    {drug:<16} {bar} (×{count})")
print(f"{'='*58}\n")