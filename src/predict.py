"""
src/predict.py
==============
Command-line prediction script.
Loads a processed .h5ad file, runs the full pipeline,
and prints the recommended drug treatment sequence.

Usage:
    python src/predict.py --input data/pbmc_processed.h5ad
    python src/predict.py --input path/to/any/processed.h5ad
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import anndata as ad
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from src.train import GATv2Classifier, sparse_to_edge_index

parser = argparse.ArgumentParser(description="SC-RLOT: Predict drug resistance and optimal treatment")
parser.add_argument("--input",  default="models/processed_adata_exp2.h5ad",
                    help="Path to a processed .h5ad file")
parser.add_argument("--config", default="configs/config.yaml")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

DEVICE     = torch.device("cpu")
DRUG_NAMES = ["AZD1775", "Venetoclax", "Dexamethasone", "Cytarabine", "Imatinib"]
SEED       = cfg["subsample"]["random_seed"]
np.random.seed(SEED)

print("\n" + "=" * 55)
print("  SC-RLOT: Cancer Drug Resistance Predictor")
print("=" * 55)

# ── 1. Load data ─────────────────────────────────────────────
if not os.path.exists(args.input):
    # Try exp1 fallback
    fallback = "models/processed_adata_exp1.h5ad"
    if os.path.exists(fallback):
        args.input = fallback
    else:
        print(f"\n[ERROR] Input not found: {args.input}")
        print("  Run 'python src/train.py --config configs/config.yaml' first.")
        sys.exit(1)

print(f"\n[1] Loading: {args.input}")
adata = ad.read_h5ad(args.input)
print(f"  {adata.n_obs} cells × {adata.n_vars} genes")
print(f"  Subclones: {adata.obs['subclone'].nunique()}")
print(f"  Resistant cells: {adata.obs['resistant'].sum()} "
      f"({adata.obs['resistant'].mean()*100:.1f}%)")

# ── 2. Find best model ───────────────────────────────────────
gat_path   = "models/gat_model_exp2.pt"
graph_path = "models/cell_graph_exp2.npz"
if not os.path.exists(gat_path):
    gat_path   = "models/gat_model_exp1.pt"
    graph_path = "models/cell_graph_exp1.npz"

# ── 3. Run GAT classifier ────────────────────────────────────
print(f"\n[2] Running GATv2 resistance classifier...")
adj = sp.load_npz(graph_path)
edge_index = sparse_to_edge_index(adj).to(DEVICE)
X = torch.tensor(adata.obsm["X_vae"], dtype=torch.float32).to(DEVICE)

model_gat = GATv2Classifier(
    in_dim=cfg["model"]["n_latent"],
    hidden_dim=cfg["model"]["gat_hidden_dim"],
    n_heads=cfg["model"]["gat_heads"],
).to(DEVICE)

if os.path.exists(gat_path):
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
    n_cells=("resistant", "count"),
    mean_resist=("resist_prob", "mean"),
).reset_index()
sc_summary["dominant"] = sc_summary["n_cells"] == sc_summary["n_cells"].max()
print(f"\n  Subclone resistance summary:")
for _, row in sc_summary.iterrows():
    tag = "← MOST RESISTANT" if row["mean_resist"] == sc_summary["mean_resist"].max() else ""
    print(f"    Subclone {row['subclone']}: {row['n_cells']:4d} cells | "
          f"resist_prob={row['mean_resist']:.3f}  {tag}")

# ── 4. Run PPO treatment recommender ────────────────────────
print(f"\n[3] Running PPO drug recommendation engine...")

from stable_baselines3 import PPO
from models.rl_env import CancerDrugEnv

cfg["model"]["n_subclones"] = adata.obs["subclone"].nunique()
gdsc2_csv = cfg["gdsc2"]["processed_csv"]
env = CancerDrugEnv(adata, gdsc2_csv if os.path.exists(gdsc2_csv) else None, cfg)

ppo_path = "models/ppo_model_exp2.zip"
if not os.path.exists(ppo_path):
    ppo_path = "models/ppo_model_exp1.zip"

if os.path.exists(ppo_path):
    ppo_model = PPO.load(ppo_path, device=DEVICE)
    use_random = False
else:
    print("  [WARN] No PPO model found — using random policy for demo")
    use_random = True

obs, _ = env.reset()
print(f"\n  Initial tumour state: {obs.round(3)}")
print(f"  Most resistant subclone: #{env.resistant_subclone} "
      f"(resist_prob={env.subclone_resist[env.resistant_subclone]:.3f})")
print(f"\n  Recommended treatment sequence:")
print(f"  {'Step':>4} | {'Drug Chosen':>15} | {'Resist Prop':>11} | {'Reward':>8}")
print(f"  {'-'*48}")

total_reward = 0.0
initial_resist = obs[env.resistant_subclone]

for step in range(cfg["model"]["episode_length"]):
    if use_random:
        action = env.action_space.sample()
    else:
        action, _ = ppo_model.predict(obs, deterministic=True)
        action = int(action)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    print(f"  {step+1:>4} | {info['drug_used']:>15} | "
          f"{info['resist_prop']:>11.4f} | {reward:>8.4f}")

final_resist = info["resist_prop"]
reduction = (initial_resist - final_resist) / (initial_resist + 1e-8) * 100

# Drug usage summary
drug_counts = {}
for h in env.episode_history:
    drug_counts[h["drug"]] = drug_counts.get(h["drug"], 0) + 1

print(f"\n{'='*55}")
print(f"  PREDICTION RESULT")
print(f"{'='*55}")
print(f"  Total treatment reward:      {total_reward:.4f}")
print(f"  Resistant subclone (start):  {initial_resist:.4f}")
print(f"  Resistant subclone (end):    {final_resist:.4f}")
print(f"  Resistance suppression:      {reduction:.1f}%")
print(f"\n  Drug usage in this episode:")
for drug, count in sorted(drug_counts.items(), key=lambda x: -x[1]):
    bar = "█" * count
    print(f"    {drug:<15} {bar} (×{count})")
print(f"{'='*55}\n")
