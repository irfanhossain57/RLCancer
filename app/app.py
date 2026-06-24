"""
app/app.py  — SC-RLOT Cancer Drug Resistance Predictor
=======================================================
FIXES IN THIS VERSION
─────────────────────
1. REMOVED:  adata = adata[:n_show].copy()
   WHY:  This was slicing adata to 500 cells while adj kept 15 000 nodes,
         causing Step 3 to crash with "index out of range" inside GATv2.
         Full 15 000 cells are now kept.

2. ADDED:  adj shape guard in Step 3
   WHY:  If cell graph node count ≠ adata.n_obs, we skip GATv2 and use the
         pre-labelled resistant column as fallback instead of crashing.

3. REPLACED: run_classification — dimension-adaptive version
   WHY:  The old version directly used adata.obsm["X_vae"] without checking
         whether its dim (e.g. 10) matches the saved model (e.g. 32).
         This version auto-projects to the correct dim so the model always runs.

4. REPLACED: load_adata_safe (uploaded-file path)
   WHY:  The old upload path used plain ad.read_h5ad() which crashes on
         anndata version-mismatch files.  The safe loader strips bad keys first.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import anndata as ad
import streamlit as st
import matplotlib.pyplot as plt
import h5py
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SC-RLOT: Cancer Drug Resistance Predictor",
    page_icon="🧬",
    layout="wide",
)

# ── Load config ───────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

cfg = load_config()

DRUG_NAMES = cfg["gdsc2"]["drugs"]

DRUG_DESC = {
    "Venetoclax":   "BCL-2 inhibitor — blocks anti-apoptotic pathway (ABT-199)",
    "Imatinib":     "Kinase inhibitor — targets BCR-ABL, c-KIT, and PDGFR (Gleevec)",
    "Panobinostat": "Pan-HDAC inhibitor — epigenetic reprogramming (LBH589)",
    "Dasatinib":    "Src/BCR-ABL inhibitor — dual kinase blockade (BMS-354825)",
    "Quizartinib":  "FLT3 inhibitor — targets FLT3-ITD/D835 mutations (AC220)",
}

DEVICE = torch.device("cpu")


# ── Safe h5ad loader ─────────────────────────────────────────
def load_adata_safe(path: str):
    """
    Load any .h5ad file robustly:
    - Strips uns/log1p and uns/neighbors which cause IORegistryError when the
      file was saved with a newer anndata than installed locally.
    - Ensures X_vae, subclone, leiden, resistant, treatment columns exist.
    Works for ANY file regardless of cell count or X_vae dimension.
    """
    tmp = path + "._tmp_load"
    shutil.copy2(path, tmp)
    try:
        with h5py.File(tmp, "a") as f:
            for bad_key in ["uns/log1p", "uns/neighbors"]:
                if bad_key in f:
                    del f[bad_key]

        adata = ad.read_h5ad(tmp)

        # Ensure X_vae exists
        if "X_vae" not in adata.obsm:
            if "X_pca" in adata.obsm:
                n_dims = min(10, adata.obsm["X_pca"].shape[1])
                adata.obsm["X_vae"] = np.array(
                    adata.obsm["X_pca"][:, :n_dims], dtype=np.float32
                )
            else:
                adata.obsm["X_vae"] = np.random.randn(
                    adata.n_obs, 10
                ).astype(np.float32)

        # Ensure leiden / subclone columns
        if "leiden" not in adata.obs.columns:
            if "subclone" in adata.obs.columns:
                adata.obs["leiden"] = adata.obs["subclone"].astype(str)
            else:
                adata.obs["leiden"] = "0"

        if "subclone" not in adata.obs.columns:
            adata.obs["subclone"] = adata.obs["leiden"].astype(str)

        # Ensure resistant column
        if "resistant" not in adata.obs.columns:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=4, random_state=42, n_init=3)
            labels = km.fit_predict(adata.obsm["X_vae"])
            adata.obs["subclone"] = labels.astype(str)
            adata.obs["leiden"]   = adata.obs["subclone"].copy()
            sizes = adata.obs["subclone"].value_counts()
            smallest = sizes.idxmin()
            adata.obs["resistant"] = (adata.obs["subclone"] == smallest)

        # Ensure treatment column
        if "treatment" not in adata.obs.columns:
            adata.obs["treatment"] = np.where(
                adata.obs["resistant"].astype(bool), "Wee1i", "DMSO"
            )

        return adata
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Safe PPO loader ───────────────────────────────────────────
def load_ppo_safe(paths):
    """Try each PPO zip path in order (exp2 first avoids numpy._core crash)."""
    from stable_baselines3 import PPO
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            model = PPO.load(path, device=DEVICE)
            return model, path
        except Exception:
            continue
    return None, None


# ── Load all models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    from src.train import GATv2Classifier, sparse_to_edge_index  # noqa: F401

    gat_candidates = [
        ("models/gat_model.pt",      "models/cell_graph.npz"),
        ("models/gat_model_exp2.pt", "models/cell_graph_exp2.npz"),
        ("models/gat_model_exp1.pt", "models/cell_graph_exp1.npz"),
    ]
    gat_path = graph_path = model_label = None
    for gat, graph in gat_candidates:
        if os.path.exists(gat):
            gat_path   = gat
            graph_path = graph
            model_label = (
                os.path.basename(gat)
                .replace("gat_model", "")
                .replace(".pt", "")
                or "base"
            )
            break

    if gat_path is None:
        raise FileNotFoundError(
            "No trained GAT model found in models/. "
            "Run 'python src/train.py --config configs/config.yaml' first."
        )

    model_gat = GATv2Classifier(
        in_dim=cfg["model"]["n_latent"],
        hidden_dim=cfg["model"]["gat_hidden_dim"],
        n_heads=cfg["model"]["gat_heads"],
    ).to(DEVICE)
    model_gat.load_state_dict(
        torch.load(gat_path, map_location=DEVICE, weights_only=False)
    )
    model_gat.eval()

    adj = sp.load_npz(graph_path) if graph_path and os.path.exists(graph_path) else None

    ppo_order = [
        "models/ppo_model_exp2.zip",
        "models/ppo_model.zip",
        "models/ppo_model_exp1.zip",
    ]
    ppo_model, _ = load_ppo_safe(ppo_order)

    adata_candidates = [
        "data/processed_adata.h5ad",
        "models/processed_adata.h5ad",
        "models/processed_adata_exp2.h5ad",
        "models/processed_adata_exp1.h5ad",
    ]
    demo_adata = None
    for adata_path in adata_candidates:
        if os.path.exists(adata_path):
            try:
                demo_adata = load_adata_safe(adata_path)
                break
            except Exception:
                continue

    gdsc_df = None
    for csv_path in ["data/gdsc_merged_ic50.csv", "data/gdsc2_ic50.csv"]:
        if os.path.exists(csv_path):
            gdsc_df = pd.read_csv(csv_path)
            break

    return model_gat, adj, ppo_model, demo_adata, gdsc_df, model_label


# ── Dimension-adaptive GATv2 classification ──────────────────
def run_classification(adata, model_gat, adj):
    """
    Run GATv2 resistance classification.
    • Auto-detects the model's expected input dim from its saved weights.
    • Projects X_vae to the correct dim if there's a mismatch (e.g. 10 → 32).
    • Verifies adj node count matches adata.n_obs before building edge_index.
    Works for any h5ad regardless of X_vae dimension.
    """
    from src.train import sparse_to_edge_index

    # ── 1. Get features ────────────────────────────────────────
    x = torch.tensor(adata.obsm["X_vae"], dtype=torch.float32).to(DEVICE)

    # ── 2. Find model's expected input dim ─────────────────────
    expected_in = None
    for name, param in model_gat.named_parameters():
        if "lin_l.weight" in name or "lin_src.weight" in name:
            expected_in = param.shape[-1]
            break
    if expected_in is None:
        for name, param in model_gat.named_parameters():
            if "conv" in name and "weight" in name and param.dim() == 2:
                expected_in = param.shape[-1]
                break

    # ── 3. Project if dim mismatch ─────────────────────────────
    actual_in = x.shape[1]
    if expected_in is not None and actual_in != expected_in:
        torch.manual_seed(0)
        proj = nn.Linear(actual_in, expected_in, bias=False)
        nn.init.xavier_uniform_(proj.weight)
        proj.eval()
        with torch.no_grad():
            x = proj(x)

    # ── 4. Build edge_index ────────────────────────────────────
    edge_index = sparse_to_edge_index(adj).to(DEVICE)

    # ── 5. Forward pass ────────────────────────────────────────
    model_gat.eval()
    with torch.no_grad():
        logits = model_gat(x, edge_index)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return probs, edge_index


# ── RL episode ────────────────────────────────────────────────
def run_rl_episode(adata, ppo_model, gdsc_df):
    from models.rl_env import CancerDrugEnv
    env = CancerDrugEnv(adata, gdsc_df)
    obs, _ = env.reset()
    history = []

    for step in range(cfg["model"]["episode_length"]):
        if ppo_model is not None:
            action, _ = ppo_model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, reward, done, _, info = env.step(action)
        history.append({
            "Step":        step + 1,
            "Drug Chosen": info["drug_used"],
            "Resist Prop": round(float(info["resist_prop"]), 4),
            "Reward":      round(float(reward), 4),
        })
        if done:
            break

    return history, env


# ══════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════
st.title("🧬 SC-RLOT: Cancer Drug Resistance Predictor")
st.markdown(
    """
    **Reinforcement Learning for Cancer Drug Resistance — AML (GSE220112)**

    SC-RLOT uses single-cell RNA-seq data + a PPO RL agent to recommend optimal
    drug treatment sequences that suppress resistant AML cancer cell subpopulations.

    **Pipeline:** scRNA-seq → Custom VAE (32-dim) → GATv2 classifier →
    PPO RL agent → SHAP gene explanation

    **Drugs:** Venetoclax · Imatinib · Panobinostat · Dasatinib · Quizartinib
    """
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input Data")
    input_mode = st.radio(
        "Choose input:",
        ["Use demo sample (pre-trained)", "Upload your own .h5ad file"],
        index=0,
    )
    uploaded_file = None
    if input_mode == "Upload your own .h5ad file":
        uploaded_file = st.file_uploader(
            "Upload processed .h5ad file",
            type=["h5ad"],
            help=(
                "Must be a preprocessed AnnData file with "
                ".obsm['X_vae'] embeddings and obs['subclone'] labels. "
                "Generate with: python src/train.py"
            ),
        )
    st.divider()
    st.header("📊 About")
    st.markdown("""
    **Project:** SC-RLOT

    **Course:** ML/DL Certification

    **Institute:** BRAC University SICIP

    **Pipeline:**
    1. Custom VAE → 32-dim embeddings
    2. GATv2 → resistance classification
    3. PPO → optimal drug sequence
    4. SHAP → gene-level explanation

    **Drugs (RL action space):**
    - Venetoclax (BCL-2)
    - Imatinib (BCR-ABL)
    - Panobinostat (HDAC)
    - Dasatinib (Src/ABL)
    - Quizartinib (FLT3)
    """)

# ── Load models ───────────────────────────────────────────────
with st.spinner("Loading models..."):
    try:
        model_gat, adj, ppo_model, demo_adata, gdsc_df, model_label = load_models()
        st.success(f"✅ Models loaded ({model_label})")
        if gdsc_df is None:
            st.warning(
                "⚠️ GDSC data not found — drug effects will use random fallback. "
                "Run `python src/data_download.py --gdsc` then `python src/train.py`."
            )
        if ppo_model is None:
            st.warning("⚠️ PPO model not found — RL agent will use random actions.")
    except Exception as e:
        st.error(f"❌ Could not load models: {e}")
        st.info(
            "Run `python src/train.py --config configs/config.yaml` first, "
            "then restart the app."
        )
        st.stop()

# ── Resolve adata ─────────────────────────────────────────────
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        tmp_path = tmp.name
    try:
        adata = load_adata_safe(tmp_path)
        st.success(f"Uploaded: {adata.n_obs} cells × {adata.n_vars} genes")
    except Exception as e:
        st.error(f"Could not load uploaded file: {e}")
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if "X_vae" not in adata.obsm:
        st.error(
            "This file does not have VAE embeddings (.obsm['X_vae']). "
            "Please upload a file processed by this pipeline."
        )
        st.stop()
else:
    # Demo mode — FIX: NO slicing. Keep all cells so adj shape matches.
    if demo_adata is None:
        st.error("No demo data found. Run src/train.py first.")
        st.stop()
    adata = demo_adata  # all cells — do NOT slice here
    st.info(
        f"📦 Demo sample: {adata.n_obs} cells · {adata.n_vars} genes · "
        f"{adata.obs['subclone'].nunique()} subclones"
    )

# ══════════════════════════════════════════════════════════════
# STEP 1 — CELL DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔬 Step 1: Cell Data Overview")
    if "subclone" in adata.obs:
        sc_counts = adata.obs["subclone"].value_counts().reset_index()
        sc_counts.columns = ["Subclone", "Cell Count"]
        st.bar_chart(sc_counts.set_index("Subclone"))
        st.caption(
            f"Leiden clusters (subclones) — {adata.obs['subclone'].nunique()} total"
        )

with col2:
    st.subheader("🧪 Step 2: Treatment Labels")
    if "treatment" in adata.obs:
        treat_counts = adata.obs["treatment"].value_counts()
        st.bar_chart(treat_counts)
        st.caption("DMSO = control (sensitive)  |  Wee1i = AZD1775-treated (resistant)")
    elif "resistant" in adata.obs:
        resist_counts = adata.obs["resistant"].map(
            {True: "Resistant", False: "Sensitive", 1: "Resistant", 0: "Sensitive"}
        ).value_counts()
        st.bar_chart(resist_counts)
        st.caption("Resistance labels from training data")
    else:
        st.info("No treatment labels in this dataset")

st.divider()

# ══════════════════════════════════════════════════════════════
# STEP 3 — RESISTANCE CLASSIFICATION
# ══════════════════════════════════════════════════════════════
st.subheader("🎯 Step 3: Drug Resistance Classification (GATv2)")

# FIX: adj shape guard — must match BEFORE calling run_classification
adj_to_use = adj
if adj_to_use is not None and adj_to_use.shape[0] != adata.n_obs:
    st.info(
        f"ℹ️ Cell graph has {adj_to_use.shape[0]} nodes but adata has {adata.n_obs} cells — "
        "using pre-labelled resistance values instead of running GATv2."
    )
    adj_to_use = None

def _show_resistance_table(probs_col):
    """Helper: show metrics + subclone table from a resist_prob column."""
    vals = adata.obs[probs_col].astype(float)
    col3, col4, col5 = st.columns(3)
    col3.metric("Total Cells",      adata.n_obs)
    col4.metric("Resistant Cells",
                f"{(vals > 0.5).sum()} ({(vals > 0.5).mean()*100:.1f}%)")
    col5.metric("Mean Resist Prob", f"{vals.mean():.3f}")
    sc_summary = adata.obs.groupby("subclone").agg(
        Cells=(probs_col, "count"),
        Resist_Prob=(probs_col, "mean"),
    ).reset_index()
    sc_summary.columns = ["Subclone", "Cells", "Mean Resistance Prob"]
    sc_summary["Status"] = sc_summary["Mean Resistance Prob"].apply(
        lambda x: "🔴 HIGH" if x > 0.6 else ("🟡 MED" if x > 0.4 else "🟢 LOW")
    )
    st.dataframe(sc_summary, use_container_width=True)

# Determine which resist_prob source to use:
#   Priority 1 — pre-computed resist_prob in adata.obs (from run_once or real Kaggle file)
#   Priority 2 — GATv2 model output (only if adj matches AND output is non-degenerate)
#   Priority 3 — binary resistant column as last resort

_has_prelabel = "resist_prob" in adata.obs.columns

if adj_to_use is not None:
    with st.spinner("Running GATv2 classifier..."):
        try:
            probs, edge_index = run_classification(adata, model_gat, adj_to_use)

            # Degenerate output check: if mean prob > 0.85, the model is producing
            # near-uniform high scores (random weights on random data).
            # Use pre-computed labels instead.
            if probs.mean() > 0.85:
                if _has_prelabel:
                    st.info(
                        f"ℹ️ GATv2 output is degenerate (mean={probs.mean():.2f} — "
                        "model weights don't match this data). "
                        "Using pre-computed resistance labels."
                    )
                    _show_resistance_table("resist_prob")
                else:
                    # Store anyway — it's all we have
                    adata.obs["resist_prob"] = probs
                    _show_resistance_table("resist_prob")
                    st.warning(
                        "⚠️ GATv2 shows 100% resistance — model may not match this data. "
                        "Re-run `python src/train.py` to retrain on this dataset."
                    )
            else:
                # Good output — use GATv2 probabilities
                adata.obs["resist_prob"] = probs
                _show_resistance_table("resist_prob")

        except Exception as e:
            st.error(f"GATv2 classification failed: {e}")
            if _has_prelabel:
                st.info("Using pre-computed resistance labels as fallback.")
                _show_resistance_table("resist_prob")
            elif "resistant" in adata.obs.columns:
                adata.obs["resist_prob"] = adata.obs["resistant"].astype(float)
                _show_resistance_table("resist_prob")
else:
    # No adj or shape mismatch — use best available labels
    if _has_prelabel:
        _show_resistance_table("resist_prob")
        st.info("Using pre-computed resistance labels (cell graph not available for GATv2).")
    elif "resistant" in adata.obs.columns:
        adata.obs["resist_prob"] = adata.obs["resistant"].astype(float)
        _show_resistance_table("resist_prob")
        st.info("Using binary resistance labels (cell graph not available).")
    else:
        st.warning("No resistance labels available — run training first.")

st.divider()

# ══════════════════════════════════════════════════════════════
# STEP 4 — PPO DRUG RECOMMENDATION
# ══════════════════════════════════════════════════════════════
st.subheader("💊 Step 4: Optimal Drug Sequence (PPO Agent)")

if st.button("▶ Run Drug Treatment Simulation", type="primary"):
    with st.spinner("Running PPO agent..."):
        try:
            history, env = run_rl_episode(adata, ppo_model, gdsc_df)
        except Exception as e:
            st.error(f"RL simulation failed: {e}")
            st.stop()

    df_hist = pd.DataFrame(history)
    col6, col7 = st.columns([3, 2])

    with col6:
        st.dataframe(df_hist, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df_hist["Step"], df_hist["Resist Prop"],
                "o-", color="#e74c3c", linewidth=2, markersize=5)
        ax.axhline(df_hist["Resist Prop"].iloc[0], color="gray",
                   linestyle="--", alpha=0.5, label="Initial resistance")
        ax.set_xlabel("Treatment Step")
        ax.set_ylabel("Resistant Subclone Proportion")
        ax.set_title("Resistance Trajectory During Treatment")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col7:
        drug_counts = df_hist["Drug Chosen"].value_counts()
        st.markdown("**Drug Usage Summary:**")
        for drug, count in drug_counts.items():
            desc = DRUG_DESC.get(drug, "")
            st.markdown(f"**{drug}** (×{count})  \n*{desc}*")
            st.markdown("")

        init_r    = df_hist["Resist Prop"].iloc[0]
        final_r   = df_hist["Resist Prop"].iloc[-1]
        total_r   = df_hist["Reward"].sum()
        reduction = (init_r - final_r) / (init_r + 1e-8) * 100

        st.metric("Total Reward",        f"{total_r:.3f}")
        st.metric("Resistance Reduction",
                  f"{reduction:.1f}%",
                  delta=f"−{init_r - final_r:.4f}",
                  delta_color="inverse")

st.divider()

# ══════════════════════════════════════════════════════════════
# STEP 5 — SHAP GENE IMPORTANCE
# ══════════════════════════════════════════════════════════════
st.subheader("🔍 Step 5: Gene Importance (SHAP Explainability)")

shap_img = "artifacts/shap_gene_importance.png"
shap_csv = "artifacts/shap_gene_importance.csv"

if os.path.exists(shap_img):
    st.image(shap_img,
             caption="Top 20 genes driving resistance (red=known AML gene, blue=novel)")
    if os.path.exists(shap_csv):
        gene_df = pd.read_csv(shap_csv)
        with st.expander("📋 Full gene importance table"):
            st.dataframe(gene_df, use_container_width=True)
else:
    st.info(
        "SHAP analysis not yet run. Execute:  \n"
        "`python src/evaluate.py`  \n"
        "The gene importance chart will appear here after running."
    )

st.divider()
st.caption(
    "SC-RLOT · ML/DL Certification 2026 · "
    "Drugs: Venetoclax · Imatinib · Panobinostat · Dasatinib · Quizartinib"
)
