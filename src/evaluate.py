"""
src/evaluate.py
===============
Generates evaluation results after training:
  - Confusion matrix for GATv2 classifier
  - SHAP gene importance plot
  - PPO vs random comparison bar chart
  - Saves all to artifacts/

CHANGES FROM ORIGINAL:
  - load_best_experiment() looks for files WITHOUT _exp2/_exp1 suffix first
    (matches Kaggle notebook output: gat_model.pt, cell_graph.npz, etc.)
    then falls back to _exp1 suffix
  - KNOWN_RESISTANCE_GENES updated for AML + new drug biology
    (WEE1/CHEK1 removed — those were AZD1775 specific;
     FLT3/MCL1/HDAC2/NF-κB genes added for the 5 new drugs)
  - plot_ppo_comparison() updated metric key names (no suffix)

Run:
    python src/evaluate.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import anndata as ad
import shap
import mlflow
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from src.train import GATv2Classifier, sparse_to_edge_index

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

MODE   = cfg["mode"]
DEVICE = torch.device("cpu")
SEED   = cfg["subsample"]["random_seed"]
os.makedirs("artifacts", exist_ok=True)
np.random.seed(SEED)

# CHANGE: Updated known gene set for AML biology and new 5 drugs
# Removed: WEE1, CHEK1, CHEK2, BRCA1 (those were AZD1775 / WEE1-inhibitor specific)
# Added:   FLT3, MCL1, HDAC2, RELA (NF-κB), BCL2L11 for new drug targets
KNOWN_RESISTANCE_GENES = {
    # BCL-2 family (Venetoclax target pathway)
    "BCL2", "MCL1", "BCL2L1", "BCL2L11", "BAX", "BAK1",
    # FLT3 pathway (Quizartinib target)
    "FLT3", "STAT5A", "STAT5B", "PIK3CA",
    # HDAC pathway (Panobinostat target)
    "HDAC1", "HDAC2", "HDAC3", "EP300",
    # BCR-ABL / Src pathway (Imatinib + Dasatinib targets)
    "ABL1", "SRC", "KIT", "PDGFRA",
    # General AML resistance genes
    "TP53", "MYC", "CDKN1A", "CDKN2A", "MDM2",
    # NF-κB / inflammatory (AML drug resistance)
    "RELA", "NFKB1", "IKBKB",
    # Cell cycle / proliferation
    "CDK4", "CDK6", "E2F1", "PCNA",
    # PBMC cell-type markers (for local pipeline test)
    "CD3D", "CD8A", "NKG7", "GNLY", "MS4A1", "CD14",
}

DRUG_NAMES = cfg["gdsc2"]["drugs"]


# ══════════════════════════════════════════════════════════════
# LOAD BEST EXPERIMENT
# ══════════════════════════════════════════════════════════════
def load_best_experiment():
    """
    CHANGE: Look for files WITHOUT suffix first (Kaggle notebook output),
    then fall back to _exp1 suffix (local baseline run).

    File search order:
        models/gat_model.pt          ← Kaggle / exp2 (preferred)
        models/gat_model_exp1.pt     ← local baseline fallback
    """
    candidates = [
        # (h5ad, graph, gat, label)
        ("models/processed_adata.h5ad",
         "models/cell_graph.npz",
         "models/gat_model.pt",
         "tuned (Kaggle/exp2)"),
        ("models/processed_adata_exp2.h5ad",
         "models/cell_graph_exp2.npz",
         "models/gat_model_exp2.pt",
         "_exp2"),
        ("models/processed_adata_exp1.h5ad",
         "models/cell_graph_exp1.npz",
         "models/gat_model_exp1.pt",
         "_exp1"),
    ]

    h5ad_path = graph_path = gat_path = label = None
    for h5ad, graph, gat, lbl in candidates:
        if os.path.exists(h5ad) and os.path.exists(gat):
            h5ad_path  = h5ad
            graph_path = graph
            gat_path   = gat
            label      = lbl
            break

    if h5ad_path is None:
        print("[ERROR] No trained model found. Run src/train.py first.")
        sys.exit(1)

    print(f"[LOAD] Using experiment: {label}")
    print(f"       adata : {h5ad_path}")
    print(f"       graph : {graph_path}")
    print(f"       GAT   : {gat_path}")

    adata      = ad.read_h5ad(h5ad_path)
    adj        = sp.load_npz(graph_path)
    edge_index = sparse_to_edge_index(adj).to(DEVICE)

    model_gat = GATv2Classifier(
        in_dim=cfg["model"]["n_latent"],
        hidden_dim=cfg["model"]["gat_hidden_dim"],
        n_heads=cfg["model"]["gat_heads"],
    ).to(DEVICE)
    model_gat.load_state_dict(torch.load(gat_path, map_location=DEVICE))
    model_gat.eval()

    return adata, model_gat, edge_index, label


# ══════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════
def plot_confusion_matrix(adata, model_gat, edge_index):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    X      = torch.tensor(adata.obsm["X_vae"], dtype=torch.float32).to(DEVICE)
    y_true = adata.obs["resistant"].values.astype(int)

    with torch.no_grad():
        logits = model_gat(X, edge_index)
        y_pred = logits.argmax(1).cpu().numpy()

    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sensitive", "Resistant"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("GATv2 Drug Resistance Classification", fontsize=12)
    plt.tight_layout()

    path = "artifacts/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")
    return path


# ══════════════════════════════════════════════════════════════
# SHAP GENE IMPORTANCE
# ══════════════════════════════════════════════════════════════
def plot_shap(adata, model_gat, edge_index):
    """
    SHAP gene importance via KernelExplainer on VAE latent space.
    Maps latent dimension importances back to genes via correlation.
    """
    print("[SHAP] Computing gene importance...")
    X_lat = adata.obsm["X_vae"]
    n_bg  = min(30, adata.n_obs)
    bg    = X_lat[np.random.choice(adata.n_obs, n_bg, replace=False)]

    resist_idx = np.where(adata.obs["resistant"].values == 1)[0]
    exp_idx    = resist_idx[:min(30, len(resist_idx))]
    if len(exp_idx) < 5:
        exp_idx = np.random.choice(adata.n_obs, 30, replace=False)
    exp_X = X_lat[exp_idx]

    def predict_fn(latent_np):
        with torch.no_grad():
            t      = torch.tensor(latent_np, dtype=torch.float32).to(DEVICE)
            logits = model_gat(t, edge_index)
            return F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, bg[:10])
    shap_vals = explainer.shap_values(exp_X[:20], nsamples=50)
    shap_arr  = np.array(shap_vals)

    # Map latent dimensions → genes via correlation
    X_gene = (adata.layers["lognorm"].toarray()
               if sp.issparse(adata.layers["lognorm"])
               else np.array(adata.layers["lognorm"]))

    mean_shap = np.mean(np.abs(shap_arr), axis=0)
    gene_imp  = np.zeros(X_gene.shape[1])
    for i in range(X_lat.shape[1]):
        corr = np.corrcoef(X_gene.T, X_lat[:, i])[:X_gene.shape[1], X_gene.shape[1]]
        gene_imp += np.abs(np.nan_to_num(corr)) * mean_shap[i]

    gene_df = pd.DataFrame({
        "gene":       adata.var_names.tolist(),
        "importance": gene_imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    gene_df.to_csv("artifacts/shap_gene_importance.csv", index=False)

    # Overlap with known AML resistance genes (CHANGE: updated gene set)
    top_genes = set(gene_df.head(20)["gene"].str.upper())
    known     = {g.upper() for g in KNOWN_RESISTANCE_GENES}
    overlap   = top_genes & known
    pct       = len(overlap) / 20 * 100
    print(f"  Known gene overlap (top-20): {len(overlap)}/20 = {pct:.1f}%")
    mlflow.log_metric("shap_known_gene_overlap_pct", pct)

    # Plot
    top_df = gene_df.head(20)
    colors = ["#e74c3c" if g.upper() in known else "#3498db"
              for g in top_df["gene"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top_df)),
            top_df["importance"][::-1].values,
            color=colors[::-1])
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df["gene"][::-1].values, fontsize=9)
    ax.set_xlabel("SHAP Importance (→ resistance)", fontsize=11)
    ax.set_title(
        f"Top 20 Genes Driving Drug Resistance\n"
        f"(red=known AML resistance gene, blue=novel | {pct:.0f}% overlap)",
        fontsize=11,
    )
    red_patch  = mpatches.Patch(color="#e74c3c", label="Known resistance gene")
    blue_patch = mpatches.Patch(color="#3498db", label="Novel candidate")
    ax.legend(handles=[red_patch, blue_patch])
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = "artifacts/shap_gene_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")
    return path, gene_df, pct


# ══════════════════════════════════════════════════════════════
# PPO COMPARISON
# ══════════════════════════════════════════════════════════════
def plot_ppo_comparison():
    """
    CHANGE: Updated metric key lookup to match new run_suffix scheme.
    Exp2 (tuned) uses no suffix: baseline_mean_reward, ppo_mean_reward
    Exp1 (baseline) uses _exp1 suffix.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
        if exp is None:
            print("  [SKIP] MLflow experiment not found")
            return None
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])

        results = []
        for run in runs:
            m     = run.data.metrics
            rname = run.info.run_name or run.info.run_id[:8]
            # CHANGE: check both suffixed and un-suffixed metric keys
            for sfx in ["", "_exp1", "_exp2"]:
                br = m.get(f"baseline_mean_reward{sfx}")
                pr = m.get(f"ppo_mean_reward{sfx}")
                if br is not None and pr is not None:
                    results.append({
                        "run":      rname + (sfx if sfx else " (tuned)"),
                        "baseline": br,
                        "ppo":      pr,
                    })
                    break

        if not results:
            print("  [SKIP] No PPO metrics in MLflow yet — run train.py first")
            return None

        fig, axes = plt.subplots(1, len(results),
                                  figsize=(6 * len(results), 5), squeeze=False)
        for i, r in enumerate(results):
            ax = axes[0][i]
            ax.bar(["Random", "PPO"], [r["baseline"], r["ppo"]],
                   color=["#e74c3c", "#2ecc71"], width=0.5)
            ax.set_title(r["run"], fontsize=10)
            ax.set_ylabel("Mean Episode Reward")
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("PPO vs Random Drug Selection — AML Treatment", fontsize=12)
        plt.tight_layout()

        path = "artifacts/ppo_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {path}")
        return path

    except Exception as e:
        print(f"  [WARN] Could not plot PPO comparison: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    adata, model_gat, edge_index, label = load_best_experiment()

    with mlflow.start_run(run_name="evaluation"):
        p1 = plot_confusion_matrix(adata, model_gat, edge_index)
        p2, gene_df, pct = plot_shap(adata, model_gat, edge_index)
        p3 = plot_ppo_comparison()

        for p in [p1, p2]:
            if p: mlflow.log_artifact(p)
        if p3: mlflow.log_artifact(p3)
        mlflow.log_artifact("artifacts/shap_gene_importance.csv")

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"  Artifacts saved to: artifacts/")
    for f in sorted(os.listdir("artifacts")):
        if not f.startswith("."): print(f"    {f}")

    print(f"\n  Top 5 resistance genes:")
    for _, row in gene_df.head(5).iterrows():
        tag = "KNOWN" if row["gene"].upper() in {g.upper() for g in KNOWN_RESISTANCE_GENES} else "novel"
        print(f"    [{tag:5s}] {row['gene']}  (score={row['importance']:.4f})")

    print(f"\n  Known gene overlap: {pct:.1f}%")
    print(f"\n  Next: python src/predict.py")


if __name__ == "__main__":
    main()
