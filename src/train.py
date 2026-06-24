"""
src/train.py
============
Main training script. Runs TWO MLflow experiments.

CHANGES FROM ORIGINAL:
  - Custom PyTorch VAE replaces scVI (matches Kaggle notebook exactly)
  - Drug list: Venetoclax, Imatinib, Panobinostat, Dasatinib, Quizartinib
  - build_gdsc_csv() now merges GDSC1 + GDSC2 (needed for all 5 drugs)
  - File save paths:
      Exp 1 → suffix "_exp1"  (baseline, fewer epochs)
      Exp 2 → suffix ""       (tuned, matches Kaggle output filenames)
  - CancerDrugEnv called with DataFrame, not CSV path
  - GDSC2 filename: 15Oct19 → 27Oct23

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --mode local
    python src/train.py --config configs/config.yaml --mode kaggle
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
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad
import scanpy as sc
import mlflow
import mlflow.pytorch
import warnings
warnings.filterwarnings("ignore")

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--mode",   default=None,
                    help="Override config mode: 'local' or 'kaggle'")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

if args.mode:
    cfg["mode"] = args.mode

MODE      = cfg["mode"]
SEED      = cfg["subsample"]["random_seed"]
MAX_CELLS = (cfg["subsample"]["max_cells_local"]
             if MODE == "local"
             else cfg["subsample"]["max_cells_kaggle"])
MAX_GENES = cfg["subsample"]["max_genes"]

# CHANGE: drug list now read from config (Venetoclax, Imatinib, Panobinostat,
#         Dasatinib, Quizartinib) instead of the old AZD1775-based list
DRUG_NAMES = cfg["gdsc2"]["drugs"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== SC-RLOT TRAINING | mode={MODE} | device={DEVICE} ===\n")
print(f"  Drugs: {DRUG_NAMES}\n")


# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
def load_data():
    if MODE == "local":
        h5_path = cfg["pbmc"]["local_path"]
        print(f"[LOAD] PBMC Multiome 3K: {h5_path}")
        adata = sc.read_10x_h5(h5_path)
        if "feature_types" in adata.var.columns:
            adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
            print(f"  RNA-only: {adata.shape[1]} genes")
        adata.obs["treatment"] = "healthy"
        adata.obs["condition"] = "control"
        adata.obs["resistant"] = 0
        print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
        return adata

    else:  # kaggle — GSE220112
        import gc
        dmso_path  = cfg["gse220112"]["dmso_h5"]
        wee1i_path = cfg["gse220112"]["wee1i_h5"]

        print(f"[LOAD] DMSO:  {dmso_path}")
        adata_dmso = sc.read_10x_h5(dmso_path)
        if "feature_types" in adata_dmso.var.columns:
            adata_dmso = adata_dmso[:, adata_dmso.var["feature_types"] == "Gene Expression"].copy()
        adata_dmso.obs["treatment"] = "DMSO"
        adata_dmso.obs["condition"] = "control"
        adata_dmso.obs["resistant"] = 0
        adata_dmso.obs_names = [f"DMSO_{bc}" for bc in adata_dmso.obs_names]
        adata_dmso.var_names_make_unique()
        print(f"  DMSO: {adata_dmso.n_obs} cells")

        print(f"[LOAD] Wee1i: {wee1i_path}")
        adata_wee1i = sc.read_10x_h5(wee1i_path)
        if "feature_types" in adata_wee1i.var.columns:
            adata_wee1i = adata_wee1i[:, adata_wee1i.var["feature_types"] == "Gene Expression"].copy()
        adata_wee1i.obs["treatment"] = "Wee1i_AZD1775"
        adata_wee1i.obs["condition"] = "treated"
        adata_wee1i.obs["resistant"] = 1
        adata_wee1i.obs_names = [f"Wee1i_{bc}" for bc in adata_wee1i.obs_names]
        adata_wee1i.var_names_make_unique()
        print(f"  Wee1i: {adata_wee1i.n_obs} cells")

        adata = ad.concat([adata_dmso, adata_wee1i], join="inner", label="sample")
        del adata_dmso, adata_wee1i; gc.collect()
        print(f"  Combined: {adata.n_obs} cells × {adata.n_vars} genes")
        return adata


# ══════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESS
# ══════════════════════════════════════════════════════════════
def preprocess(adata, max_cells=None, max_genes=None):
    if max_cells is None: max_cells = MAX_CELLS
    if max_genes is None: max_genes = MAX_GENES
    print(f"\n[PREPROCESS] {adata.n_obs}c × {adata.n_vars}g → target {max_cells}c, {max_genes}g")

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After QC: {adata.n_obs}c × {adata.n_vars}g")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=max_genes,
                                 flavor="seurat_v3", layer="counts")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  After HVG: {adata.n_obs}c × {adata.n_vars}g")

    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=SEED)
        print(f"  After subsample: {adata.n_obs}c")

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=cfg["model"]["k_neighbors"],
                    n_pcs=30, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    sc.tl.leiden(adata, resolution=0.5, random_state=SEED, key_added="subclone")
    n_clusters = adata.obs["subclone"].nunique()
    print(f"  Leiden subclones: {n_clusters}")

    if MODE == "local":
        np.random.seed(SEED)
        adata.obs["resistant"] = (np.random.rand(adata.n_obs) > 0.5).astype(int)
        print("  [LOCAL] Random resistance labels (pipeline test only)")
    else:
        n_r = adata.obs["resistant"].sum()
        print(f"  [KAGGLE] Resistant(Wee1i): {n_r}, Sensitive(DMSO): {adata.n_obs - n_r}")

    adata.obs["resistant_label"] = adata.obs["resistant"].map(
        {0: "sensitive", 1: "resistant"})
    return adata, n_clusters


# ══════════════════════════════════════════════════════════════
# STEP 3 — CELL GRAPH  (for GATv2)
# ══════════════════════════════════════════════════════════════
def build_graph(adata, out_path):
    from sklearn.neighbors import kneighbors_graph
    k   = cfg["model"]["k_neighbors"]
    A   = kneighbors_graph(adata.obsm["X_pca"], n_neighbors=k,
                            mode="connectivity", metric="cosine",
                            include_self=False)
    A   = (A + A.T).astype(bool).astype(float)
    sp.save_npz(out_path, A)
    print(f"  Graph: {A.shape[0]} nodes, {A.nnz // 2} edges → {out_path}")
    return A


# ══════════════════════════════════════════════════════════════
# STEP 4 — CUSTOM PYTORCH VAE
# CHANGE: replaces scVI with the custom VAE from the Kaggle notebook
# ══════════════════════════════════════════════════════════════
class VAE(nn.Module):
    """
    Custom Variational Autoencoder matching the Kaggle notebook exactly.
    Saved as models/custom_vae.pt — a dict with keys:
        model_state, input_dim, latent_dim
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decode(z)
        return x_recon, mu, log_var


def vae_loss(x_recon, x, mu, log_var):
    """ELBO = reconstruction (MSE) + KL divergence."""
    recon = F.mse_loss(x_recon, x, reduction="sum")
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl


def train_vae(adata, n_epochs, run_suffix=""):
    """
    Train custom PyTorch VAE on log-normalised gene expression.
    Stores 32-dim latent vectors in adata.obsm['X_vae'].
    Saves weights to models/custom_vae{run_suffix}.pt
    """
    import scipy.sparse as sp_check

    # Extract log-normalised matrix
    X_raw = adata.layers["lognorm"]
    if sp_check.issparse(X_raw):
        X_raw = X_raw.toarray()
    X_raw = X_raw.astype(np.float32)

    input_dim  = X_raw.shape[1]
    hidden_dim = cfg["model"]["n_hidden"]
    latent_dim = cfg["model"]["n_latent"]

    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(dataset, batch_size=128, shuffle=True)

    vae = VAE(input_dim, hidden_dim, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    print(f"  VAE: input_dim={input_dim}, latent={latent_dim}, epochs={n_epochs}")
    vae.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            x_recon, mu, log_var = vae(batch)
            loss = vae_loss(x_recon, batch, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader.dataset)
        mlflow.log_metric(f"vae_loss{run_suffix}", avg, step=epoch)
        if epoch % max(1, n_epochs // 5) == 0:
            print(f"  VAE epoch {epoch:3d}/{n_epochs} | loss={avg:.4f}")

    # Extract latent representations
    vae.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_raw, dtype=torch.float32).to(DEVICE)
        mu, _ = vae.encode(X_t)
        Z = mu.cpu().numpy()

    adata.obsm["X_vae"] = Z
    print(f"  VAE embeddings: {Z.shape}")

    # Save model — same dict format as Kaggle notebook
    os.makedirs("models", exist_ok=True)
    save_path = f"models/custom_vae{run_suffix}.pt"
    torch.save({
        "model_state": vae.state_dict(),
        "input_dim":   input_dim,
        "latent_dim":  latent_dim,
    }, save_path)
    print(f"  Saved: {save_path}")
    return adata, vae


# ══════════════════════════════════════════════════════════════
# STEP 5 — GATv2 CLASSIFIER
# ══════════════════════════════════════════════════════════════
class GATv2Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_heads, n_classes=2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.conv1      = GATv2Conv(in_dim, hidden_dim, heads=n_heads,
                                     concat=True, dropout=0.3)
        self.conv2      = GATv2Conv(hidden_dim * n_heads, hidden_dim,
                                     heads=n_heads, concat=False, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout    = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index)); x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index)); x = self.dropout(x)
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
    y = torch.tensor(adata.obs["resistant"].values.astype(int),
                     dtype=torch.long).to(DEVICE)

    n   = adata.n_obs
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED,
        stratify=adata.obs["resistant"].values,
    )
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_idx]     = True

    model = GATv2Classifier(
        in_dim=cfg["model"]["n_latent"],
        hidden_dim=cfg["model"]["gat_hidden_dim"],
        n_heads=cfg["model"]["gat_heads"],
    ).to(DEVICE)

    counts  = np.bincount(adata.obs["resistant"].values.astype(int))
    weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]],
                            dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_auc, best_state = 0.0, None

    for epoch in range(1, n_epochs + 1):
        model.train(); optimizer.zero_grad()
        logits = model(X, edge_index)
        loss   = criterion(logits[train_mask], y[train_mask])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_v = model(X, edge_index)
            probs_v  = F.softmax(logits_v[val_mask], dim=1)[:, 1].cpu().numpy()
            preds_v  = logits_v[val_mask].argmax(1).cpu().numpy()
            y_v      = y[val_mask].cpu().numpy()

        val_loss = criterion(logits_v[val_mask], y[val_mask]).item()
        scheduler.step(val_loss)

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
            best_auc  = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % max(1, n_epochs // 5) == 0:
            print(f"  GAT {epoch:3d}/{n_epochs} | loss={loss.item():.4f} "
                  f"| AUC={auc:.4f} | F1={f1:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best GAT AUC: {best_auc:.4f}")
    mlflow.log_metric(f"gat_best_auc{run_suffix}", best_auc)

    # CHANGE: no suffix in final model path for exp2 (matches Kaggle output)
    gat_path = f"models/gat_model{run_suffix}.pt"
    torch.save(model.state_dict(), gat_path)

    model.eval()
    with torch.no_grad():
        embeds    = model.embed(X, edge_index).cpu().numpy()
        probs_all = F.softmax(model(X, edge_index), dim=1)[:, 1].cpu().numpy()

    np.save(f"models/cell_embeddings{run_suffix}.npy", embeds)
    adata.obs[f"resist_prob{run_suffix}"] = probs_all

    # For the final experiment (no suffix), also write to the main resist_prob column
    if run_suffix == "":
        adata.obs["resist_prob"] = probs_all

    return model, best_auc, adata


# ══════════════════════════════════════════════════════════════
# STEP 6 — GDSC1 + GDSC2 MERGE
# CHANGE: now reads BOTH GDSC1 and GDSC2 to cover all 5 drugs
# ══════════════════════════════════════════════════════════════
def build_gdsc_csv():
    """
    Merge GDSC1 and GDSC2 drug sensitivity data into one CSV.
    - Venetoclax, Dasatinib       → GDSC2
    - Imatinib, Panobinostat,
      Quizartinib                 → GDSC1
    Saves to data/gdsc_merged_ic50.csv
    """
    out = "data/gdsc_merged_ic50.csv"
    if os.path.exists(out):
        print(f"  [SKIP] Merged GDSC CSV already exists: {out}")
        return out

    leukemia_lines = cfg["gdsc2"]["leukemia_lines"]

    # Drug aliases → canonical name
    aliases = {
        "Venetoclax":    ["Venetoclax", "ABT-199"],
        "Imatinib":      ["Imatinib", "Gleevec", "STI571"],
        "Panobinostat":  ["Panobinostat", "LBH589"],
        "Dasatinib":     ["Dasatinib", "BMS-354825"],
        "Quizartinib":   ["Quizartinib", "AC220"],
    }

    frames = []

    # ── GDSC2 ────────────────────────────────────────────────
    gdsc2_path = cfg["gdsc2"]["local_path"]
    if os.path.exists(gdsc2_path):
        print(f"  [GDSC2] Loading {gdsc2_path}...")
        g2 = pd.read_excel(gdsc2_path)
        for canonical, alts in aliases.items():
            subset = g2[g2["DRUG_NAME"].isin(alts)].copy()
            if len(subset):
                subset["DRUG_CANONICAL"] = canonical
                subset["SOURCE"] = "GDSC2"
                frames.append(subset[["CELL_LINE_NAME", "DRUG_NAME",
                                       "DRUG_CANONICAL", "LN_IC50", "SOURCE"]])
        print(f"    {len(g2)} rows total")
    else:
        print(f"  [WARN] GDSC2 not found at {gdsc2_path} — run data_download.py first")

    # ── GDSC1 ────────────────────────────────────────────────
    gdsc1_path = cfg["gdsc1"]["local_path"]
    if os.path.exists(gdsc1_path):
        print(f"  [GDSC1] Loading {gdsc1_path}...")
        g1 = pd.read_excel(gdsc1_path)
        for canonical, alts in aliases.items():
            subset = g1[g1["DRUG_NAME"].isin(alts)].copy()
            if len(subset):
                subset["DRUG_CANONICAL"] = canonical
                subset["SOURCE"] = "GDSC1"
                frames.append(subset[["CELL_LINE_NAME", "DRUG_NAME",
                                       "DRUG_CANONICAL", "LN_IC50", "SOURCE"]])
        print(f"    {len(g1)} rows total")
    else:
        print(f"  [WARN] GDSC1 not found at {gdsc1_path} — run data_download.py first")

    if not frames:
        print("  [ERROR] No GDSC data found — using random drug effects in RL env")
        return None

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate: if a drug appears in both GDSC1 and GDSC2, keep GDSC2
    df = df.sort_values("SOURCE").drop_duplicates(
        subset=["CELL_LINE_NAME", "DRUG_CANONICAL"], keep="last")

    # Scale LN_IC50 to [0, 1] per drug
    for drug in df["DRUG_CANONICAL"].unique():
        mask = df["DRUG_CANONICAL"] == drug
        vals = df.loc[mask, "LN_IC50"]
        df.loc[mask, "IC50_scaled"] = (
            (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        )

    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  Saved merged GDSC CSV: {out} ({len(df)} rows, "
          f"{df['DRUG_CANONICAL'].nunique()} drugs)")
    return out


def load_gdsc_df(csv_path):
    """Load merged GDSC CSV into a DataFrame for the RL environment."""
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# ══════════════════════════════════════════════════════════════
# STEP 7 — PPO AGENT
# ══════════════════════════════════════════════════════════════
def train_ppo(adata, gdsc_df, run_suffix="", timesteps=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from models.rl_env import CancerDrugEnv

    if timesteps is None:
        timesteps = cfg["model"]["ppo_timesteps"]

    cfg["model"]["n_subclones"] = adata.obs["subclone"].nunique()

    # CHANGE: pass DataFrame directly, not CSV path
    env = Monitor(CancerDrugEnv(adata, gdsc_df))

    # ── Baseline: random policy ──────────────────────────────
    raw_env = CancerDrugEnv(adata, gdsc_df)
    baseline_rewards = []
    for _ in range(50):
        obs, _ = raw_env.reset()
        ep_r   = 0.0
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

    print(f"  Training PPO for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # ── Evaluate PPO ─────────────────────────────────────────
    ppo_rewards, final_resist = [], []
    for _ in range(50):
        obs, _ = raw_env.reset()
        ep_r   = 0.0
        for _ in range(cfg["model"]["episode_length"]):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = raw_env.step(int(a))
            ep_r += r
        ppo_rewards.append(ep_r)
        final_resist.append(info["resist_prop"])

    ppo_mean        = float(np.mean(ppo_rewards))
    ppo_resist_mean = float(np.mean(final_resist))
    improvement     = (ppo_mean - baseline_mean) / (abs(baseline_mean) + 1e-8) * 100

    mlflow.log_metrics({
        f"ppo_mean_reward{run_suffix}":       ppo_mean,
        f"ppo_mean_resist_prop{run_suffix}":  ppo_resist_mean,
        f"reward_improvement_pct{run_suffix}": improvement,
    })
    print(f"  PPO reward: {ppo_mean:.4f} | Improvement: {improvement:.1f}%")

    # CHANGE: no suffix for exp2 matches Kaggle filename
    ppo_path = f"models/ppo_model{run_suffix}"
    model.save(ppo_path)
    return model, ppo_mean, improvement


# ══════════════════════════════════════════════════════════════
# RUN ONE EXPERIMENT
# ══════════════════════════════════════════════════════════════
def run_experiment(run_name, vae_epochs, gat_epochs, ppo_steps, run_suffix):
    """
    Full pipeline: preprocess → VAE → GATv2 → PPO → log to MLflow.

    run_suffix:
        "_exp1"  → baseline  (files: gat_model_exp1.pt, ppo_model_exp1.zip ...)
        ""       → tuned     (files: gat_model.pt, ppo_model.zip ...  ← Kaggle match)
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {run_name}")
    print(f"  VAE={vae_epochs}ep | GAT={gat_epochs}ep | PPO={ppo_steps}steps")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name=run_name) as run:
        print(f"[MLFLOW] Run: {run.info.run_id}")

        mlflow.log_params({
            "mode":       MODE,
            "vae_epochs": vae_epochs,
            "gat_epochs": gat_epochs,
            "ppo_steps":  ppo_steps,
            "max_cells":  MAX_CELLS,
            "max_genes":  MAX_GENES,
            "n_latent":   cfg["model"]["n_latent"],
            "gat_heads":  cfg["model"]["gat_heads"],
            "drugs":      str(DRUG_NAMES),
        })

        # Load + preprocess
        adata_raw        = load_data()
        adata, n_clones  = preprocess(adata_raw)
        cfg["model"]["n_subclones"] = n_clones
        mlflow.log_param("n_subclones", n_clones)

        # Cell graph
        os.makedirs("models", exist_ok=True)
        # CHANGE: graph path — no suffix for exp2 to match Kaggle
        graph_path = f"models/cell_graph{run_suffix}.npz"
        adj = build_graph(adata, graph_path)

        # GDSC1 + GDSC2 merge
        gdsc_csv = build_gdsc_csv()
        gdsc_df  = load_gdsc_df(gdsc_csv)

        # VAE (custom PyTorch)
        print(f"\n[VAE] Training {vae_epochs} epochs...")
        adata, _ = train_vae(adata, vae_epochs, run_suffix)

        # GATv2
        print(f"\n[GAT] Training {gat_epochs} epochs...")
        _, best_auc, adata = train_gat(adata, adj, gat_epochs, run_suffix)

        # PPO
        print(f"\n[PPO] Training {ppo_steps} steps...")
        _, ppo_mean, improvement = train_ppo(adata, gdsc_df, run_suffix, ppo_steps)

        # Save processed adata
        # CHANGE: no suffix for exp2 to match Kaggle filename
        h5ad_path = f"models/processed_adata{run_suffix}.h5ad"
        adata.write_h5ad(h5ad_path)
        mlflow.log_artifact(h5ad_path)

        print(f"\n[DONE] {run_name}:")
        print(f"  GAT AUC:         {best_auc:.4f}")
        print(f"  PPO reward:      {ppo_mean:.4f}")
        print(f"  vs random:       +{improvement:.1f}%")
        print(f"  MLflow run:      {run.info.run_id}")
        return best_auc, ppo_mean


# ══════════════════════════════════════════════════════════════
# MAIN — TWO EXPERIMENTS
# ══════════════════════════════════════════════════════════════
def main():
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    if MODE == "local":
        vae1, gat1, ppo1 = 5,  10, 500
        vae2, gat2, ppo2 = 10, 15, 1000
    else:
        vae1, gat1, ppo1 = 30,  25, 10000
        vae2, gat2, ppo2 = 100, 80, 50000

    # ── Experiment 1: Baseline  (files get _exp1 suffix) ────
    auc1, ppo1_r = run_experiment(
        run_name=cfg["mlflow"]["run1_name"],
        vae_epochs=vae1, gat_epochs=gat1, ppo_steps=ppo1,
        run_suffix="_exp1",
    )

    # ── Experiment 2: Tuned  (files get NO suffix = Kaggle match) ──
    auc2, ppo2_r = run_experiment(
        run_name=cfg["mlflow"]["run2_name"],
        vae_epochs=vae2, gat_epochs=gat2, ppo_steps=ppo2,
        run_suffix="",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Exp 1 (baseline): GAT AUC={auc1:.4f} | PPO={ppo1_r:.4f}")
    print(f"  Exp 2 (tuned):    GAT AUC={auc2:.4f} | PPO={ppo2_r:.4f}")
    print(f"\n  View results:  mlflow ui --port 5000")
    print(f"  Evaluate:      python src/evaluate.py")


if __name__ == "__main__":
    main()
