"""
src/train.py
============
Main training script. Runs TWO MLflow experiments as required
by the assignment (at least 2 runs with different settings).

Usage:
    python src/train.py --config configs/config.yaml

What it does:
    Experiment 1 (baseline):  VAE 20 epochs, GAT 20 epochs, PPO 5k steps
    Experiment 2 (tuned):     VAE 50 epochs, GAT 40 epochs, PPO 15k steps
    Both logged to MLflow → compare in UI at localhost:5000

On Kaggle: this same script runs with larger epoch/step counts
from the Kaggle notebook cell.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import anndata as ad
import scanpy as sc
import mlflow
import mlflow.pytorch
import warnings
warnings.filterwarnings("ignore")

# ── args ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--mode",   default=None,
                    help="Override config mode: 'local' or 'kaggle'")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

if args.mode:
    cfg["mode"] = args.mode

MODE = cfg["mode"]
SEED = cfg["subsample"]["random_seed"]
MAX_CELLS = (cfg["subsample"]["max_cells_local"]
             if MODE == "local"
             else cfg["subsample"]["max_cells_kaggle"])
MAX_GENES = cfg["subsample"]["max_genes"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== SC-RLOT TRAINING | mode={MODE} | device={DEVICE} ===\n")


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    """
    Load the correct dataset depending on mode.
    LOCAL  → PBMC Multiome 3K (.h5 file, 38.8 MB)
    KAGGLE → GSE220112 (two .h5 files: DMSO + Wee1i)
    """
    if MODE == "local":
        h5_path = cfg["pbmc"]["local_path"]
        print(f"[LOAD] PBMC Multiome 3K from: {h5_path}")
        adata = sc.read_10x_h5(h5_path)

        # This .h5 has both RNA and ATAC — keep RNA only
        if "feature_types" in adata.var.columns:
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
            print(f"  RNA-only subset: {adata.shape[1]} genes")

        adata.obs["treatment"] = "healthy"
        adata.obs["condition"] = "control"
        print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
        return adata

    else:  # kaggle — GSE220112
        dmso_path  = cfg["gse220112"]["dmso_h5"]
        wee1i_path = cfg["gse220112"]["wee1i_h5"]

        print(f"[LOAD] GSE220112 DMSO:  {dmso_path}")
        adata_dmso = sc.read_10x_h5(dmso_path)
        if "feature_types" in adata_dmso.var.columns:
            adata_dmso = adata_dmso[:, adata_dmso.var["feature_types"] == "Gene Expression"].copy()
        adata_dmso.obs["treatment"] = "DMSO"
        adata_dmso.obs["condition"] = "control"
        adata_dmso.obs_names = [f"DMSO_{bc}" for bc in adata_dmso.obs_names]
        print(f"  DMSO: {adata_dmso.n_obs} cells × {adata_dmso.n_vars} genes")

        print(f"[LOAD] GSE220112 Wee1i: {wee1i_path}")
        adata_wee1i = sc.read_10x_h5(wee1i_path)
        if "feature_types" in adata_wee1i.var.columns:
            adata_wee1i = adata_wee1i[:, adata_wee1i.var["feature_types"] == "Gene Expression"].copy()
        adata_wee1i.obs["treatment"] = "Wee1i_AZD1775"
        adata_wee1i.obs["condition"] = "treated"
        adata_wee1i.obs_names = [f"Wee1i_{bc}" for bc in adata_wee1i.obs_names]
        print(f"  Wee1i: {adata_wee1i.n_obs} cells × {adata_wee1i.n_vars} genes")

        # Concatenate: inner join (keep genes in both)
        adata = ad.concat([adata_dmso, adata_wee1i], join="inner", label="sample")
        print(f"  Combined: {adata.n_obs} cells × {adata.n_vars} genes")
        return adata


# ─────────────────────────────────────────────────────────────
# STEP 2: PREPROCESS
# ─────────────────────────────────────────────────────────────
def preprocess(adata, max_cells=None, max_genes=None):
    """QC → Normalize → HVG → Subsample → PCA → Leiden clusters."""
    if max_cells is None: max_cells = MAX_CELLS
    if max_genes is None: max_genes = MAX_GENES

    print(f"\n[PREPROCESS] {adata.n_obs} cells × {adata.n_vars} genes → target {max_cells} cells, {max_genes} genes")

    # ── QC ───────────────────────────────────────────────────
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After QC: {adata.n_obs} cells × {adata.n_vars} genes")

    # ── Normalize ────────────────────────────────────────────
    adata.layers["counts"] = adata.X.copy()   # raw counts for scVI
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    # ── HVG ─────────────────────────────────────────────────
    sc.pp.highly_variable_genes(adata, n_top_genes=max_genes,
                                 flavor="seurat_v3", layer="counts")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  After HVG: {adata.n_obs} cells × {adata.n_vars} genes")

    # ── Subsample ────────────────────────────────────────────
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=SEED)
        print(f"  After subsample: {adata.n_obs} cells")

    # ── PCA + neighbors + UMAP + Leiden ─────────────────────
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=cfg["model"]["k_neighbors"],
                    n_pcs=30, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(adata, resolution=0.5, random_state=SEED, key_added="subclone")
    n_clusters = adata.obs["subclone"].nunique()
    print(f"  Leiden subclones: {n_clusters}")

    # ── Resistance labels ────────────────────────────────────
    if MODE == "local":
        # PBMC: random labels (pipeline test only — NOT real biology)
        np.random.seed(SEED)
        adata.obs["resistant"] = (np.random.rand(adata.n_obs) > 0.5).astype(int)
        print("  [LOCAL] Random resistance labels (pipeline validation only)")
    else:
        # GSE220112: Wee1i-treated survivors = resistant subclone
        adata.obs["resistant"] = (adata.obs["treatment"] == "Wee1i_AZD1775").astype(int)
        n_r = adata.obs["resistant"].sum()
        print(f"  [KAGGLE] Resistant (Wee1i): {n_r}, Sensitive (DMSO): {adata.n_obs - n_r}")

    adata.obs["resistant_label"] = adata.obs["resistant"].map({0: "sensitive", 1: "resistant"})
    return adata, n_clusters


# ─────────────────────────────────────────────────────────────
# STEP 3: BUILD CELL GRAPH (for GATv2)
# ─────────────────────────────────────────────────────────────
def build_graph(adata, out_path):
    from sklearn.neighbors import kneighbors_graph
    k = cfg["model"]["k_neighbors"]
    X_pca = adata.obsm["X_pca"]
    A = kneighbors_graph(X_pca, n_neighbors=k, mode="connectivity",
                          metric="cosine", include_self=False)
    A = (A + A.T).astype(bool).astype(float)
    sp.save_npz(out_path, A)
    print(f"  Graph: {A.shape[0]} nodes, {A.nnz//2} edges → {out_path}")
    return A


# ─────────────────────────────────────────────────────────────
# STEP 4: VAE ENCODER (scVI)
# ─────────────────────────────────────────────────────────────
def train_vae(adata, n_epochs, run_suffix=""):
    import scvi
    scvi.settings.seed = SEED
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(
        adata,
        n_latent=cfg["model"]["n_latent"],
        n_layers=cfg["model"]["n_layers"],
        n_hidden=cfg["model"]["n_hidden"],
        dropout_rate=0.1,
    )
    kwargs = dict(
        max_epochs=n_epochs,
        early_stopping=True,
        early_stopping_patience=5,
        train_size=0.9,
        batch_size=128,
        plan_kwargs={"lr": 1e-3},
    )
    kwargs["accelerator"] = "gpu" if DEVICE.type == "cuda" else "cpu"
    model.train(**kwargs)

    adata.obsm["X_vae"] = model.get_latent_representation()
    print(f"  VAE embeddings: {adata.obsm['X_vae'].shape}")

    # Log VAE training history
    if hasattr(model, "history") and "elbo_train" in model.history:
        for i, v in enumerate(model.history["elbo_train"]["elbo_train"]):
            mlflow.log_metric(f"vae_train_elbo{run_suffix}", v, step=i)

    # Save model
    save_path = f"models/scvi_vae{run_suffix}"
    model.save(save_path, overwrite=True)
    return adata, model


# ─────────────────────────────────────────────────────────────
# STEP 5: GATv2 CLASSIFIER
# ─────────────────────────────────────────────────────────────
class GATv2Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_heads, n_classes=2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=n_heads,
                                concat=True, dropout=0.3)
        self.conv2 = GATv2Conv(hidden_dim * n_heads, hidden_dim,
                                heads=n_heads, concat=False, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index));  x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index));  x = self.dropout(x)
        return self.classifier(x)

    def embed(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return F.elu(self.conv2(x, edge_index))


def sparse_to_edge_index(adj):
    coo = adj.tocoo()
    return torch.stack([
        torch.tensor(coo.row, dtype=torch.long),
        torch.tensor(coo.col, dtype=torch.long),
    ], dim=0)


def train_gat(adata, adj, n_epochs, run_suffix=""):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score

    edge_index = sparse_to_edge_index(adj).to(DEVICE)
    X = torch.tensor(adata.obsm["X_vae"], dtype=torch.float32).to(DEVICE)
    y = torch.tensor(adata.obs["resistant"].values.astype(int), dtype=torch.long).to(DEVICE)

    n = adata.n_obs
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED,
        stratify=adata.obs["resistant"].values
    )
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_idx]     = True

    model_gat = GATv2Classifier(
        in_dim=cfg["model"]["n_latent"],
        hidden_dim=cfg["model"]["gat_hidden_dim"],
        n_heads=cfg["model"]["gat_heads"],
    ).to(DEVICE)

    counts = np.bincount(adata.obs["resistant"].values.astype(int))
    weights = torch.tensor([1.0/counts[0], 1.0/counts[1]], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model_gat.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_auc = 0.0; best_state = None

    for epoch in range(1, n_epochs + 1):
        model_gat.train(); optimizer.zero_grad()
        logits = model_gat(X, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward(); optimizer.step()

        model_gat.eval()
        with torch.no_grad():
            logits_v = model_gat(X, edge_index)
            probs_v  = F.softmax(logits_v[val_mask], dim=1)[:,1].cpu().numpy()
            preds_v  = logits_v[val_mask].argmax(1).cpu().numpy()
            y_v      = y[val_mask].cpu().numpy()

        val_loss = criterion(logits_v[val_mask], y[val_mask]).item()
        try:    auc = roc_auc_score(y_v, probs_v)
        except: auc = 0.5
        f1 = f1_score(y_v, preds_v, average="weighted", zero_division=0)

        mlflow.log_metrics({
            f"gat_train_loss{run_suffix}": loss.item(),
            f"gat_val_loss{run_suffix}":   val_loss,
            f"gat_val_auc{run_suffix}":    auc,
            f"gat_val_f1{run_suffix}":     f1,
        }, step=epoch)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model_gat.state_dict().items()}

        if epoch % max(1, n_epochs // 5) == 0:
            print(f"  GAT Epoch {epoch:3d}/{n_epochs} | "
                  f"loss={loss.item():.4f} | AUC={auc:.4f} | F1={f1:.4f}")

    model_gat.load_state_dict(best_state)
    print(f"  Best GAT AUC: {best_auc:.4f}")
    mlflow.log_metric(f"gat_best_auc{run_suffix}", best_auc)

    # Save model
    gat_path = f"models/gat_model{run_suffix}.pt"
    torch.save(model_gat.state_dict(), gat_path)

    # Save cell embeddings (used by RL env)
    model_gat.eval()
    with torch.no_grad():
        embeds = model_gat.embed(X, edge_index).cpu().numpy()
    np.save(f"models/cell_embeddings{run_suffix}.npy", embeds)

    # Save resistance probabilities on all cells
    with torch.no_grad():
        probs_all = F.softmax(model_gat(X, edge_index), dim=1)[:,1].cpu().numpy()
    adata.obs[f"resist_prob{run_suffix}"] = probs_all

    return model_gat, best_auc, adata


# ─────────────────────────────────────────────────────────────
# STEP 6: RL ENVIRONMENT + PPO
# ─────────────────────────────────────────────────────────────
def build_rl_env(adata, gdsc2_csv_path):
    """Build the Gymnasium environment. Imported from models/rl_env.py."""
    sys.path.insert(0, ".")
    from models.rl_env import CancerDrugEnv
    return CancerDrugEnv(adata, gdsc2_csv_path, cfg)


def build_gdsc2_csv():
    """Extract leukemia IC50 values from GDSC2 Excel into a clean CSV."""
    path = cfg["gdsc2"]["local_path"]
    out  = cfg["gdsc2"]["processed_csv"]
    if os.path.exists(out):
        print(f"  [SKIP] GDSC2 CSV already exists: {out}")
        return

    print(f"  [GDSC2] Reading {path}...")
    gdsc2 = pd.read_excel(path)

    # Column names in the 15Oct19 version:
    # CELL_LINE_NAME, DRUG_NAME, LN_IC50, AUC
    drugs   = cfg["gdsc2"]["drugs"]
    lines   = cfg["gdsc2"]["leukemia_lines"]

    # Filter for our drugs (handle aliases)
    aliases = {
        "AZD1775":       ["AZD1775", "adavosertib", "MK-1775"],
        "Venetoclax":    ["Venetoclax", "ABT-199"],
        "Dexamethasone": ["Dexamethasone"],
        "Cytarabine":    ["Cytarabine", "Ara-C"],
        "Imatinib":      ["Imatinib", "Gleevec", "STI571"],
    }

    rows = []
    for canonical, alts in aliases.items():
        subset = gdsc2[gdsc2["DRUG_NAME"].isin(alts)].copy()
        subset["DRUG_CANONICAL"] = canonical
        rows.append(subset)

    df = pd.concat(rows, ignore_index=True)
    print(f"  Found {len(df)} rows for 5 target drugs")

    # Scale LN_IC50 to [0,1] per drug
    for drug in df["DRUG_CANONICAL"].unique():
        mask = df["DRUG_CANONICAL"] == drug
        vals = df.loc[mask, "LN_IC50"]
        df.loc[mask, "IC50_scaled"] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  Saved GDSC2 CSV: {out} ({len(df)} rows)")


def train_ppo(adata, run_suffix="", timesteps=None):
    """Train PPO agent and compare against random baseline."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    if timesteps is None:
        timesteps = cfg["model"]["ppo_timesteps"]

    # GDSC2 reward table
    gdsc2_csv = cfg["gdsc2"]["processed_csv"]
    if not os.path.exists(gdsc2_csv):
        build_gdsc2_csv()

    # Force n_subclones to match actual data
    cfg["model"]["n_subclones"] = adata.obs["subclone"].nunique()

    env = Monitor(build_rl_env(adata, gdsc2_csv))

    # ── Baseline: random policy ──────────────────────────────
    from models.rl_env import CancerDrugEnv
    raw_env = CancerDrugEnv(adata, gdsc2_csv, cfg)
    baseline_rewards = []
    for _ in range(50):
        obs, _ = raw_env.reset()
        ep_r = 0.0
        for _ in range(cfg["model"]["episode_length"]):
            a = raw_env.action_space.sample()
            obs, r, done, _, _ = raw_env.step(a)
            ep_r += r
        baseline_rewards.append(ep_r)
    baseline_mean = float(np.mean(baseline_rewards))
    mlflow.log_metric(f"baseline_mean_reward{run_suffix}", baseline_mean)
    print(f"  Random baseline reward: {baseline_mean:.4f}")

    # ── PPO training ─────────────────────────────────────────
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=128,
                batch_size=32, n_epochs=10, gamma=cfg["model"]["gamma"],
                ent_coef=0.01, verbose=0, seed=SEED)

    ep_rewards = []

    class _Cb:
        def __init__(self): self.n_calls = 0
        def __call__(self, *a, **kw): pass

    print(f"  Training PPO for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # ── Evaluate PPO policy ──────────────────────────────────
    ppo_rewards = []
    final_resist = []
    for _ in range(50):
        obs, _ = raw_env.reset()
        ep_r = 0.0
        for _ in range(cfg["model"]["episode_length"]):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = raw_env.step(int(a))
            ep_r += r
        ppo_rewards.append(ep_r)
        final_resist.append(info["resist_prop"])

    ppo_mean = float(np.mean(ppo_rewards))
    ppo_resist_mean = float(np.mean(final_resist))
    improvement = (ppo_mean - baseline_mean) / (abs(baseline_mean) + 1e-8) * 100

    mlflow.log_metrics({
        f"ppo_mean_reward{run_suffix}":       ppo_mean,
        f"ppo_mean_resist_prop{run_suffix}":  ppo_resist_mean,
        f"reward_improvement_pct{run_suffix}": improvement,
    })
    print(f"  PPO reward: {ppo_mean:.4f} | Improvement: {improvement:.1f}%")

    ppo_path = f"models/ppo_model{run_suffix}"
    model.save(ppo_path)
    return model, ppo_mean, improvement


# ─────────────────────────────────────────────────────────────
# RUN ONE FULL EXPERIMENT
# ─────────────────────────────────────────────────────────────
def run_experiment(exp_name, run_name, vae_epochs, gat_epochs, ppo_steps, run_suffix):
    """Full pipeline: preprocess → VAE → GAT → PPO → log all to MLflow."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {run_name}")
    print(f"  VAE epochs={vae_epochs}, GAT epochs={gat_epochs}, PPO steps={ppo_steps}")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[MLFLOW] Run ID: {run.info.run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "mode":        MODE,
            "vae_epochs":  vae_epochs,
            "gat_epochs":  gat_epochs,
            "ppo_steps":   ppo_steps,
            "max_cells":   MAX_CELLS,
            "max_genes":   MAX_GENES,
            "n_latent":    cfg["model"]["n_latent"],
            "gat_heads":   cfg["model"]["gat_heads"],
            "k_neighbors": cfg["model"]["k_neighbors"],
        })

        # Load + preprocess
        adata_raw = load_data()
        adata, n_clusters = preprocess(adata_raw)
        cfg["model"]["n_subclones"] = n_clusters
        mlflow.log_param("n_subclones", n_clusters)

        # Build cell graph
        os.makedirs("models", exist_ok=True)
        graph_path = f"models/cell_graph{run_suffix}.npz"
        adj = build_graph(adata, graph_path)

        # Build GDSC2 table
        build_gdsc2_csv()

        # VAE
        print(f"\n[VAE] Training {vae_epochs} epochs...")
        adata, _ = train_vae(adata, vae_epochs, run_suffix)

        # GAT
        print(f"\n[GAT] Training {gat_epochs} epochs...")
        model_gat, best_auc, adata = train_gat(adata, adj, gat_epochs, run_suffix)

        # PPO
        print(f"\n[PPO] Training {ppo_steps} steps...")
        _, ppo_mean, improvement = train_ppo(adata, run_suffix, ppo_steps)

        # Save final adata
        h5ad_path = f"models/processed_adata{run_suffix}.h5ad"
        adata.write_h5ad(h5ad_path)
        mlflow.log_artifact(h5ad_path)

        print(f"\n[DONE] Run '{run_name}' complete:")
        print(f"  GAT AUC: {best_auc:.4f}")
        print(f"  PPO mean reward: {ppo_mean:.4f}")
        print(f"  Improvement over random: {improvement:.1f}%")
        print(f"  MLflow run: {run.info.run_id}\n")

        return best_auc, ppo_mean


# ─────────────────────────────────────────────────────────────
# MAIN — TWO EXPERIMENTS (assignment requires ≥2)
# ─────────────────────────────────────────────────────────────
def main():
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    if MODE == "local":
        # LOCAL: fast runs to verify code
        vae1, gat1, ppo1 = 5,  10, 500
        vae2, gat2, ppo2 = 10, 15, 1000
    else:
        # KAGGLE: real training runs
        vae1, gat1, ppo1 = 30, 25, 10000
        vae2, gat2, ppo2 = 80, 50, 30000

    # ── EXPERIMENT 1: Baseline ──────────────────────────────
    auc1, ppo1_r = run_experiment(
        exp_name=cfg["mlflow"]["experiment_name"],
        run_name=cfg["mlflow"]["run1_name"],
        vae_epochs=vae1, gat_epochs=gat1, ppo_steps=ppo1,
        run_suffix="_exp1",
    )

    # ── EXPERIMENT 2: Tuned ─────────────────────────────────
    auc2, ppo2_r = run_experiment(
        exp_name=cfg["mlflow"]["experiment_name"],
        run_name=cfg["mlflow"]["run2_name"],
        vae_epochs=vae2, gat_epochs=gat2, ppo_steps=ppo2,
        run_suffix="_exp2",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — BOTH EXPERIMENTS")
    print("=" * 60)
    print(f"  Exp 1 (baseline): GAT AUC={auc1:.4f} | PPO reward={ppo1_r:.4f}")
    print(f"  Exp 2 (tuned):    GAT AUC={auc2:.4f} | PPO reward={ppo2_r:.4f}")
    print(f"\n  View in MLflow:")
    print(f"    mlflow ui --host 0.0.0.0 --port 5000")
    print(f"    Open: http://localhost:5000")
    print(f"\n  Next step:")
    print(f"    python src/evaluate.py")


if __name__ == "__main__":
    main()
