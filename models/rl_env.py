"""
models/rl_env.py  — SC-RLOT Cancer Drug Environment
====================================================
FIX IN THIS VERSION
───────────────────
Step 4 crashed with:
  "Unexpected observation shape (4,) for Box environment,
   please use (9,) or (n_env, 9)"

Root cause: The saved PPO model was trained with n_subclones=9
(real Kaggle Leiden clustering on 15k GSE220112 cells).
The environment dynamically sets observation_space shape to
however many subclones are in the uploaded adata — if that file
has only 4 subclones, obs shape is (4,) and the PPO rejects it.

Fix: observation_space is always fixed at OBS_DIM=9.
The actual state vector is zero-padded to OBS_DIM before
returning from reset() and step().  This way ANY adata with
≤9 subclones works with the saved PPO model.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Drug definitions (must match config.yaml + train.py) ─────
DRUG_NAMES = [
    "Venetoclax",    # action 0 — BCL-2 inhibitor
    "Imatinib",      # action 1 — BCR-ABL/c-KIT/PDGFR inhibitor
    "Panobinostat",  # action 2 — pan-HDAC inhibitor
    "Dasatinib",     # action 3 — Src/BCR-ABL inhibitor
    "Quizartinib",   # action 4 — FLT3 inhibitor
]

TOXICITY = np.array([0.4, 0.3, 0.5, 0.35, 0.4])

N_DRUGS        = len(DRUG_NAMES)   # 5
EPISODE_LENGTH = 10                 # treatment steps per episode

# FIX: observation space is always this size — matches the saved PPO model
# (PPO was trained on 15k Kaggle cells → 9 Leiden clusters → obs_dim=9)
OBS_DIM = 9


class CancerDrugEnv(gym.Env):
    """
    Simulated cancer treatment environment.

    State  : proportion vector, zero-padded to OBS_DIM=9, sums to ≤1
    Action : integer 0-4, one drug per step
    Reward : reduction in resistant subclone − toxicity penalty − burden term

    Constructor
    -----------
    adata     : AnnData with obs['subclone'] and obs['resist_prob'] columns
    gdsc_df   : combined GDSC1+GDSC2 DataFrame (None = random fallback)
    episode_length : int
    seed      : int
    """

    metadata = {"render_modes": []}

    def __init__(self, adata, gdsc_df=None, episode_length=EPISODE_LENGTH, seed=42):
        super().__init__()

        self.adata          = adata
        self.gdsc_df        = gdsc_df
        self.episode_length = episode_length
        self.rng            = np.random.default_rng(seed)

        self._build_subclone_profiles()
        self._build_drug_effects()

        # FIX: always OBS_DIM=9 — matches saved PPO regardless of actual subclone count
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_DRUGS)

        self.state           = None
        self.step_count      = 0
        self.episode_history = []

    # ── Internal helpers ──────────────────────────────────────

    def _pad_obs(self, state):
        """
        Pad or truncate state vector to OBS_DIM.
        state shape: (n_subclones,) → (OBS_DIM,)
        Extra slots are zero (no cells in those phantom subclones).
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        n   = min(len(state), OBS_DIM)
        obs[:n] = state[:n]
        return obs

    def _build_subclone_profiles(self):
        """Compute initial proportions and resistance scores per subclone."""
        subclone_col  = self.adata.obs["subclone"].astype(str)
        unique_clones = sorted(subclone_col.unique())
        self.n_subclones = len(unique_clones)
        self.clone_ids   = unique_clones

        total = len(self.adata)
        self.initial_proportions = np.array(
            [(subclone_col == c).sum() / total for c in unique_clones],
            dtype=np.float32,
        )

        if "resist_prob" in self.adata.obs.columns:
            self.subclone_resist = np.array(
                [self.adata.obs.loc[subclone_col == c, "resist_prob"].mean()
                 for c in unique_clones],
                dtype=np.float32,
            )
        else:
            self.subclone_resist = self.rng.random(self.n_subclones).astype(np.float32)

        self.resistant_subclone = int(np.argmax(self.subclone_resist))

    def _build_drug_effects(self):
        """Build kill-rate matrix: drug_effect[d, s] = fraction killed per step."""
        self.drug_effect = np.zeros((N_DRUGS, self.n_subclones), dtype=np.float32)

        if self.gdsc_df is not None and len(self.gdsc_df) > 0:
            for d_idx, drug in enumerate(DRUG_NAMES):
                drug_rows = self.gdsc_df[self.gdsc_df["DRUG_CANONICAL"] == drug]
                if len(drug_rows) == 0:
                    base_kill = self.rng.uniform(0.10, 0.25, self.n_subclones)
                else:
                    mean_ic50   = drug_rows["IC50_scaled"].mean()
                    sensitivity = float(np.clip(1.0 - mean_ic50, 0.05, 0.60))
                    base_kill   = np.full(self.n_subclones, sensitivity, dtype=np.float32)

                for s in range(self.n_subclones):
                    resist_factor = 1.0 - self.subclone_resist[s] * 0.6
                    bk = base_kill[s] if np.ndim(base_kill) > 0 else float(base_kill)
                    self.drug_effect[d_idx, s] = float(bk * resist_factor)
        else:
            self.drug_effect = self.rng.uniform(
                0.05, 0.30, (N_DRUGS, self.n_subclones)
            ).astype(np.float32)

        # Resistance-specific biology penalties
        self.drug_effect[0, self.resistant_subclone] *= 0.70  # Venetoclax — MCL1 escape
        self.drug_effect[4, self.resistant_subclone] *= 0.60  # Quizartinib — FLT3-WT

    # ── Gymnasium API ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        noise = self.rng.normal(0, 0.02, self.n_subclones).astype(np.float32)
        state = np.clip(self.initial_proportions + noise, 0.0, 1.0)
        state /= state.sum() + 1e-8

        self.state           = state.copy()
        self.step_count      = 0
        self.episode_history = []

        # FIX: return padded obs so shape is always OBS_DIM
        return self._pad_obs(self.state), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_resist = float(self.state[self.resistant_subclone])
        kill_rates  = self.drug_effect[action]

        new_state = self.state * (1.0 - kill_rates)

        # Resistant subclone regrowth (Darwinian pressure)
        regrowth = 0.03 * self.subclone_resist[self.resistant_subclone]
        new_state[self.resistant_subclone] += regrowth

        total = new_state.sum()
        self.state = (new_state / total if total > 1e-8
                      else self.initial_proportions.copy()).astype(np.float32)

        curr_resist = float(self.state[self.resistant_subclone])

        # Reward components
        r_suppress = (prev_resist - curr_resist) * 10.0
        r_toxicity = -float(TOXICITY[action]) * 0.5
        r_burden   = -float(np.dot(self.state, self.subclone_resist)) * 0.5
        reward     = float(r_suppress + r_toxicity + r_burden)

        self.step_count += 1
        self.episode_history.append({
            "step":        self.step_count,
            "drug":        DRUG_NAMES[action],
            "drug_idx":    int(action),
            "resist_prop": curr_resist,
            "reward":      reward,
        })

        terminated = self.step_count >= self.episode_length
        info = {
            "drug_used":   DRUG_NAMES[action],
            "resist_prop": curr_resist,
            "step":        self.step_count,
        }

        # FIX: return padded obs so shape is always OBS_DIM
        return self._pad_obs(self.state), reward, terminated, False, info

    def render(self):
        print(f"  Step {self.step_count}/{self.episode_length} "
              f"| resistant prop: {self.state[self.resistant_subclone]:.4f}")

    def close(self):
        pass