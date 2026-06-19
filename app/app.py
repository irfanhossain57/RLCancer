"""
app/app.py
==========
Streamlit prediction app — runs on port 8501 (Docker requirement).

Features:
  - Upload a .h5ad file OR use the demo sample
  - Shows resistance classification per cell
  - Shows subclone composition
  - Runs PPO agent and shows drug recommendation sequence
  - Shows SHAP gene importance chart

Run locally:
    streamlit run app/app.py --server.port 8501

Run via Docker:
    docker build -t sc-rlot-app:1.0 .
    docker run -p 8501:8501 sc-rlot-app:1.0
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import anndata as ad
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SC-RLOT: Cancer Drug Resistance Predictor",
    page_icon="🧬",
    layout="wide",
)

# ── Load config ──────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

cfg = load_config()

DRUG_NAMES = ["AZD1775", "Venetoclax", "Dexamethasone", "Cytarabine", "Imatinib"]
DRUG_DESC = {
    "AZD1775":       "WEE1 kinase inhibitor — targets G2/M checkpoint",
    "Venetoclax":    "BCL-2 inhibitor — blocks anti-apoptotic pathway",
    "Dexamethasone": "Corticosteroid — standard ALL chemotherapy",
    "Cytarabine":    "Nucleoside analogue — disrupts DNA replication",
    "Imatinib":      "Kinase inhibitor — targets BCR-ABL/c-KIT/PDGFR",
}
DEVICE = torch.device("cpu")


# ── Load models (cached so they don't reload on every interaction) ──
@st.cache_resource
def load_models():
    from src.train import GATv2Classifier, sparse_to_edge_index

    # Find best available model
    for suffix in ["_exp2", "_exp1", ""]:
        gat_path   = f"models/gat_model{suffix}.pt"
        graph_path = f"models/cell_graph{suffix}.npz"
        adata_path = f"models/processed_adata{suffix}.h5ad"
        if os.path.exists(gat_path):
            break

    # GAT
    model_gat = GATv2Classifier(
        in_dim=cfg["model"]["n_latent"],
        hidden_dim=cfg["model"]["gat_hidden_dim"],
        n_heads=cfg["model"]["gat_heads"],
    ).to(DEVICE)
    if os.path.exists(gat_path):
        model_gat.load_state_dict(torch.load(gat_path, map_location=DEVICE))
    model_gat.eval()

    # Graph
    adj = sp.load_npz(graph_path) if os.path.exists(graph_path) else None

    # PPO
    ppo_model = None
    for psfx in ["_exp2", "_exp1", ""]:
        ppo_path = f"models/ppo_model{psfx}.zip"
        if os.path.exists(ppo_path):
            from stable_baselines3 import PPO
            ppo_model = PPO.load(ppo_path, device=DEVICE)
            break

    # Default adata (demo sample)
    demo_adata = ad.read_h5ad(adata_path) if os.path.exists(adata_path) else None

    return model_gat, adj, ppo_model, demo_adata, suffix


def run_classification(adata, model_gat, adj):
    from src.train import sparse_to_edge_index
    edge_index = sparse_to_edge_index(adj).to(DEVICE)
    X = torch.tensor(adata.obsm["X_vae"], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model_gat(X, edge_index)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs, edge_index, X


def run_rl_episode(adata, ppo_model):
    from models.rl_env import CancerDrugEnv
    cfg["model"]["n_subclones"] = adata.obs["subclone"].nunique()
    gdsc2_csv = cfg["gdsc2"]["processed_csv"]
    env = CancerDrugEnv(adata, gdsc2_csv if os.path.exists(gdsc2_csv) else None, cfg)
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
            "Step": step + 1,
            "Drug Chosen": info["drug_used"],
            "Resist Prop": round(info["resist_prop"], 4),
            "Reward": round(reward, 4),
        })
    return history, env


# ── UI Layout ────────────────────────────────────────────────
st.title("🧬 SC-RLOT: Cancer Drug Resistance Predictor")
st.markdown(
    """
    **Reinforcement Learning for Cancer Drug Resistance**  
    *SC-RLOT uses single-cell RNA-seq data + RL to recommend optimal drug treatment sequences
    that suppress resistant cancer cell subpopulations.*

    **Dataset:** PBMC Multiome 3K (local demo) | GSE220112 ALL leukemia (real cancer results)  
    **Models:** scVI VAE encoder → GATv2 classifier → PPO RL agent
    """
)
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
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
            help="Must be a preprocessed AnnData file with .obsm['X_vae'] and obs['subclone']"
        )
    st.divider()
    st.header("📊 About")
    st.markdown("""
    **Project:** SC-RLOT Mini  
    **Course:** ML/DL Certification  
    **Institute:** BRAC University SICIP  
    **Deadline:** 25 June 2026  
    
    **Pipeline:**  
    1. VAE → 32-dim cell embeddings  
    2. GATv2 → resistance classification  
    3. PPO → optimal drug sequence  
    4. SHAP → gene-level explanation
    """)

# ── Load models ──────────────────────────────────────────────
with st.spinner("Loading models..."):
    try:
        model_gat, adj, ppo_model, demo_adata, model_suffix = load_models()
        models_loaded = True
        st.success(f"✅ Models loaded (experiment: {model_suffix})", icon="✅")
    except Exception as e:
        models_loaded = False
        st.error(f"❌ Could not load models: {e}")
        st.info("Run `python src/train.py --config configs/config.yaml` first, then restart the app.")
        st.stop()

# ── Get adata ────────────────────────────────────────────────
if uploaded_file is not None:
    import tempfile, shutil
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        tmp_path = tmp.name
    adata = ad.read_h5ad(tmp_path)
    st.success(f"Uploaded: {adata.n_obs} cells × {adata.n_vars} genes")
    if "X_vae" not in adata.obsm:
        st.error("This .h5ad file does not have VAE embeddings (.obsm['X_vae']). "
                 "Please upload a file processed by this pipeline.")
        st.stop()
else:
    if demo_adata is None:
        st.error("No demo data found. Run training first.")
        st.stop()
    adata = demo_adata
    st.info(f"📦 Demo sample: {adata.n_obs} cells (first {min(500, adata.n_obs)} shown)")
    # Use subset for display speed
    adata = adata[:min(500, adata.n_obs)].copy()

# ── Run pipeline ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔬 Step 1: Cell Data Overview")
    if "subclone" in adata.obs:
        subclone_counts = adata.obs["subclone"].value_counts().reset_index()
        subclone_counts.columns = ["Subclone", "Cell Count"]
        st.bar_chart(subclone_counts.set_index("Subclone"))
        st.caption("Leiden clusters (subclones) in the sample")

with col2:
    st.subheader("🧪 Step 2: Treatment History")
    if "treatment" in adata.obs:
        treat_counts = adata.obs["treatment"].value_counts()
        st.bar_chart(treat_counts)
        st.caption("DMSO = control (sensitive), Wee1i = AZD1775-treated (resistant)")
    else:
        st.info("No treatment labels (PBMC demo mode — labels are random)")

st.divider()

# ── Resistance Classification ─────────────────────────────────
st.subheader("🎯 Step 3: Drug Resistance Classification (GATv2)")

if adj is not None:
    with st.spinner("Running GATv2 classifier..."):
        probs, edge_index, X_tensor = run_classification(adata, model_gat, adj)
    adata.obs["resist_prob"] = probs

    col3, col4, col5 = st.columns(3)
    col3.metric("Total Cells", adata.n_obs)
    col4.metric("Resistant Cells", f"{(probs > 0.5).sum()} ({(probs > 0.5).mean()*100:.1f}%)")
    col5.metric("Mean Resist Prob", f"{probs.mean():.3f}")

    sc_summary = adata.obs.groupby("subclone").agg(
        Cells=("resistant", "count"),
        Resist_Prob=("resist_prob", "mean"),
    ).reset_index()
    sc_summary.columns = ["Subclone", "Cells", "Mean Resistance Prob"]
    sc_summary["Status"] = sc_summary["Mean Resistance Prob"].apply(
        lambda x: "🔴 HIGH" if x > 0.6 else ("🟡 MED" if x > 0.4 else "🟢 LOW")
    )
    st.dataframe(sc_summary, use_container_width=True)
else:
    st.warning("No cell graph found — run training first.")

st.divider()

# ── PPO Drug Recommendation ───────────────────────────────────
st.subheader("💊 Step 4: Optimal Drug Sequence (PPO Agent)")

if st.button("▶ Run Drug Treatment Simulation", type="primary"):
    with st.spinner("Running PPO agent..."):
        history, env = run_rl_episode(adata, ppo_model)

    df_hist = pd.DataFrame(history)
    col6, col7 = st.columns([3, 2])

    with col6:
        st.dataframe(df_hist, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df_hist["Step"], df_hist["Resist Prop"], "o-", color="#e74c3c", linewidth=2)
        ax.axhline(df_hist["Resist Prop"].iloc[0], color="gray",
                   linestyle="--", alpha=0.5, label="Initial")
        ax.set_xlabel("Treatment Step"); ax.set_ylabel("Resistant Subclone Proportion")
        ax.set_title("Resistance Trajectory During Treatment"); ax.legend()
        ax.grid(alpha=0.3); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col7:
        drug_counts = df_hist["Drug Chosen"].value_counts()
        st.markdown("**Drug Usage Summary:**")
        for drug, count in drug_counts.items():
            desc = DRUG_DESC.get(drug, "")
            st.markdown(f"**{drug}** (×{count})  \n*{desc}*")

        total_r = df_hist["Reward"].sum()
        init_r  = df_hist["Resist Prop"].iloc[0]
        final_r = df_hist["Resist Prop"].iloc[-1]
        reduction = (init_r - final_r) / (init_r + 1e-8) * 100
        st.metric("Total Reward", f"{total_r:.3f}")
        st.metric("Resistance Reduction", f"{reduction:.1f}%",
                  delta=f"-{init_r-final_r:.4f}",
                  delta_color="inverse")

st.divider()

# ── SHAP ─────────────────────────────────────────────────────
st.subheader("🔍 Step 5: Gene Importance (SHAP Explainability)")

shap_csv = "artifacts/shap_gene_importance.csv"
shap_img = "artifacts/shap_gene_importance.png"

if os.path.exists(shap_img):
    st.image(shap_img, caption="Top 20 genes driving resistance prediction (red=known, blue=novel)")
    if os.path.exists(shap_csv):
        gene_df = pd.read_csv(shap_csv)
        with st.expander("📋 Full gene importance table"):
            st.dataframe(gene_df, use_container_width=True)
else:
    st.info("SHAP analysis not yet run. Execute: `python src/evaluate.py`")
    st.caption("The SHAP plot will appear here after running evaluation.")

st.divider()
st.caption("SC-RLOT Mini | BRAC University SICIP | ML/DL Certification 2026")
