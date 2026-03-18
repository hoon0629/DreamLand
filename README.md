# Conformalized DreamerV3 + MPPI for Hazard-Aware Planetary Landing

A research framework combining a learned world model, sampling-based control,
and calibrated uncertainty quantification for hazard-aware precision landing
on planetary terrain using real NASA data.

```
NASA Terrain Data (Lunar DEM / AI4MARS)
             ↓
   Terrain Hazard Map (slope + roughness)
             ↓
   DreamerV3 World Model  ←─── train on PDG simulation rollouts
   (RSSM latent dynamics)
             ↓
   MPPI Planner  ←─── K=512 imagined rollouts via Prior RSSM
   (online receding-horizon control)
             ↓
   Conformal Prediction Layer  ←─── calibrated uncertainty on rollouts
   (nonconformity score → c_conf penalty)
             ↓
   Safe Thrust Command [Tx, Ty, Tz]
```

---

## Project Structure

```
mars_nav/
├── README.md                  ← you are here
│
├── mars_nav_pipeline.py       ← Track 1: Mars rover terrain navigation
├── mars_nav_uq_pipeline.py    ← Track 1 + UQ: MC Dropout uncertainty
├── train_ai4mars.py           ← Train DeepLabV3+ on NASA AI4MARS dataset
│
├── dreamerv3_pdg.py           ← Track 2: DreamerV3 world model for PDG
├── mppi_dreamer.py            ← Track 2 + MPPI: online planner + terrain
└── pdg_vjepa2.py              ← Track 2 alt: V-JEPA 2 world model (vision)
```

Two tracks, same research goal:

| Track | Task | Model | Data |
|-------|------|-------|------|
| 1 — Rover Navigation | Terrain segmentation + path planning | DeepLabV3+ + A* | NASA AI4MARS |
| 2 — PDG Landing | Powered descent guidance | DreamerV3 + MPPI | Simulation + Lunar DEM |

---

## Requirements

### System

- Python 3.10 or 3.11 recommended
- CUDA GPU recommended for DreamerV3 training (CPU works but is slow)
- 8 GB RAM minimum; 16 GB recommended for DreamerV3

### Install all dependencies

```bash
pip install torch torchvision
pip install numpy pillow matplotlib scipy
pip install networkx requests
pip install lightning-uq-box
pip install transformers  # for V-JEPA 2 (optional)
```

One-liner:

```bash
pip install torch torchvision numpy pillow matplotlib scipy networkx requests lightning-uq-box
```

### Verify installation

```python
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
```

---

## Track 1 — Mars Rover Terrain Navigation

Segments Mars surface imagery into terrain classes, builds a traversability
cost map, and plans a safe path using A*.

### File: `mars_nav_pipeline.py`

**What it does:**
1. Fetches a rover image from NASA API (or uses a synthetic one offline)
2. Runs DeepLabV3+ terrain segmentation → 4-class label map
3. Converts labels to traversability cost map
4. Runs A* path planning from top-left to bottom-right of the map
5. Saves a 4-panel visualization as `mars_nav_output.png`

**Quick start (no API key, no data download needed):**

```bash
python mars_nav_pipeline.py
```

This uses a synthetic Mars-like image generated offline. No internet required.

**With NASA live imagery:**

```bash
# Get a free key at https://api.nasa.gov
# Then edit the bottom of mars_nav_pipeline.py:
```

```python
results = run_pipeline(
    api_key="YOUR_KEY_HERE",
    sol=1000,                # Martian sol — Curiosity has 4000+
    camera="NAVCAM",         # NAVCAM, MAST, FHAZ, RHAZ
    planning_grid_size=80,
    output_path="mars_nav_output.png",
)
```

**Output:** `mars_nav_output.png` — 4 panels:

| Panel | Content |
|-------|---------|
| 1 | Original NASA rover image |
| 2 | Terrain segmentation overlay (soil/bedrock/sand/rock) |
| 3 | Traversability cost heatmap |
| 4 | A* planned path |

**Terrain classes (AI4MARS standard):**

| Label | Class | Traversal Cost |
|-------|-------|---------------|
| 0 | Soil | 1.0 — safe |
| 1 | Bedrock | 3.0 — moderate |
| 2 | Sand | 7.0 — risky (slippage) |
| 3 | Rock | 100.0 — obstacle |

---

### File: `mars_nav_uq_pipeline.py`

Same pipeline as above, but wraps DeepLabV3+ with **MC Dropout** via
`lightning-uq-box` to produce per-pixel epistemic uncertainty.

**Key upgrade — uncertainty-aware cost map:**

```
cost(pixel) = base_terrain_cost × (1 + α × uncertainty)
```

A pixel the model is unsure about gets elevated cost even if the predicted
class is safe. This makes the planner risk-aware, not just obstacle-aware.

**Run:**

```bash
python mars_nav_uq_pipeline.py
```

**Output:** `mars_nav_uq_output.png` — 5 panels (adds uncertainty heatmap).

**Tune uncertainty sensitivity:**

```python
results = run_pipeline(
    uncertainty_alpha=5.0,   # higher → more risk-averse
    num_mc_samples=20,        # more samples → better uncertainty estimate
    dropout_p=0.3,
)
```

---

### File: `train_ai4mars.py`

Fine-tunes DeepLabV3+ on the real NASA AI4MARS dataset for best accuracy
(~95% validation accuracy reported in literature).

**Step 1 — Download AI4MARS:**

Go to https://data.nasa.gov and search "AI4MARS". Download and unzip to:

```
ai4mars/
├── images/       ← rover .jpg images (~35K)
├── labels/       ← .png label maps (pixel values 0–3)
├── train.txt     ← list of training image IDs
└── val.txt       ← list of validation image IDs
```

**Step 2 — Train:**

```bash
python train_ai4mars.py \
  --data_dir ./ai4mars \
  --epochs 30 \
  --batch_size 4 \
  --img_size 512 \
  --output ./checkpoints
```

GPU strongly recommended. On an A100 this takes ~2 hours for 30 epochs.
On CPU this is not feasible; use Google Colab with a GPU runtime instead.

**Step 3 — Use your fine-tuned weights:**

In `mars_nav_uq_pipeline.py`, update `UncertaintyAwareSegmenter.__init__()`:

```python
# After wrapping with MCDropoutSegmentation:
ckpt = torch.load("checkpoints/deeplabv3_ai4mars_best.pth")
self.model.base_model.load_state_dict(ckpt["model"])
print(f"Loaded fine-tuned weights, mIoU={ckpt['miou']:.4f}")
```

---

## Track 2 — Powered Descent Guidance (PDG)

Learns a world model of Mars landing dynamics and uses it for online
trajectory optimization. No vision data required — pure state trajectories.

**State space:** `[x, y, z, vx, vy, vz, mass]`
**Action space:** `[Tx, Ty, Tz]` normalized thrust ∈ [-1, 1]³
**Goal:** soft pinpoint landing — position error < 5m, speed < 2 m/s

---

### File: `dreamerv3_pdg.py`

Trains a DreamerV3 world model on the Mars PDG simulation environment.

**Architecture:**

```
obs [x,y,z,vx,vy,vz,mass]
    ↓ symlog normalization
StateEncoder (MLP)
    ↓ embed_t
RSSM ←── GRU deterministic state h_t
    ├── Posterior q(z_t | h_t, embed_t)   ← training
    └── Prior    p(z_t | h_t)             ← imagination / MPPI
    ↓ [h_t, z_t]
Decoder → obs prediction
RewardHead → reward (two-hot symlog)
ContinueHead → done probability
    ↓ imagination rollout (H=15 steps, no env)
Actor → action
Critic → value (two-hot symlog) → λ-returns
```

**DreamerV3 features implemented:**
- symlog observation transforms (handles altitude 0–2000m range)
- KL balancing (α=0.8) + free bits (1 nat) — prevents posterior collapse
- Unimix categoricals (1% uniform) — stabilizes stochastic state
- Percentile return normalization (5th/95th) — stable actor training
- Two-hot symlog value and reward loss — handles sparse reward spikes
- Block GRU + RMSNorm + SiLU throughout

**Run:**

```bash
python dreamerv3_pdg.py
```

Default: 50,000 environment steps. For convergence use 200,000+:

```python
# Edit bottom of dreamerv3_pdg.py:
agent, metrics, eval_results = train_dreamerv3(
    n_env_steps=200_000,
    train_every=5,
    seed_episodes=10,
    plot=True,
)
```

**Output:** `dreamerv3_pdg_output.png` — 8 panels:
- Training: world model loss, actor/critic loss, episode return, success rate
- Evaluation: descent trajectories, landing scatter, error distribution

**Save and load a checkpoint:**

```python
# Save
torch.save(agent.state_dict(), "dreamerv3_pdg.pth")

# Load
agent = DreamerV3(DreamerConfig()).to(device)
agent.load_state_dict(torch.load("dreamerv3_pdg.pth", map_location=device))
agent.eval()
```

---

### File: `mppi_dreamer.py`

Adds an MPPI online planner on top of the trained DreamerV3 world model,
plus a terrain hazard map derived from a synthetic (or real) DEM.

**MPPI algorithm:**

```
At each real timestep t:
  1. Encode obs_t → (h_t, z_t)  via Posterior  [ONCE]
  2. Sample K=512 action perturbations ε_{k,t} ~ N(0, σ²)
  3. For each k, roll H=20 steps using Prior only  [K×H steps]
     → decode each latent → predicted state
     → accumulate cost J_k = Σ c_task + c_hazard + c_control + c_conf
  4. w_k = softmax(-normalize(J_k) / λ)
  5. U_t ← U_t + Σ_k w_k · ε_k
  6. Execute U_t[0]; shift U left (warm start)
  7. If all J_k > safety_threshold → fallback brake policy
```

**Cost function:**

| Term | Formula | Weight |
|------|---------|--------|
| `c_task` | `‖[x,y]‖² + 0.1·alt` | `w_task=1.0` |
| `c_vel` | `‖[vx,vy,vz]‖²` | `w_vel=0.5` |
| `c_control` | `‖action‖²` (fuel) | `w_control=0.05` |
| `c_hazard` | terrain hazard score at (x,y) | `w_hazard=2.0` |
| `c_conf` | conformal uncertainty penalty | `w_conf=0.0` ← enabled in Phase 5 |
| `c_terminal` | large pos+vel+crash penalty | `w_terminal=5.0` |

**Requires `dreamerv3_pdg.py` in the same directory.**

**Run (trains DreamerV3 then evaluates MPPI):**

```bash
python mppi_dreamer.py
```

This runs the full pipeline:
1. Trains DreamerV3 world model (50k steps)
2. Builds terrain hazard map (synthetic DEM with craters)
3. Evaluates MPPI vs DreamerV3 Actor baseline (30 episodes each)
4. Saves comparison plot as `mppi_dreamer_output.png`

**Output:** `mppi_dreamer_output.png` — 6 panels:
- MPPI footprint overlaid on terrain hazard map
- Descent altitude profiles (MPPI vs Actor)
- Landing scatter for both planners
- Hazard-at-touchdown histogram
- Summary metrics bar chart

**Use your own terrain DEM:**

```python
import numpy as np
dem = np.load("lunar_dem_tile.npy")         # your DEM array (N×N float32)
terrain = TerrainHazardMap(
    size=dem.shape[0],
    resolution=5.0,                          # meters per cell
    dem_array=dem,
)
```

**Tune MPPI hyperparameters:**

```python
mppi_cfg = MPPIConfig(
    K=1024,            # more samples → better plan, slower
    H=25,              # longer horizon → better foresight
    temperature=0.03,  # lower → more winner-takes-all
    w_hazard=5.0,      # higher → more hazard-averse
    safety_threshold=500.0,  # lower → fallback triggers sooner
    sigma=0.3,         # action noise standard deviation
)
```

---

### File: `pdg_vjepa2.py` (alternative)

Uses V-JEPA 2 (Meta FAIR) as the world model instead of DreamerV3.
Requires video observations (rendered frames) rather than state vectors.
Only relevant if you add a simulation renderer (AirSim / Gazebo).

```bash
# MLP proxy mode (no vision, CPU-friendly):
python pdg_vjepa2.py

# V-JEPA 2 ViT-L from HuggingFace (GPU, ~4GB VRAM):
pip install -U git+https://github.com/huggingface/transformers
# Then set mode="lite" in pdg_vjepa2.py
```

---

## Recommended Run Order (Full Project)

```bash
# Step 1: Verify environment
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

# Step 2: Track 1 — quick sanity check (no data, no GPU)
python mars_nav_pipeline.py
# → mars_nav_output.png

# Step 3: Track 1 with uncertainty
python mars_nav_uq_pipeline.py
# → mars_nav_uq_output.png

# Step 4: Track 2 — DreamerV3 training + evaluation
python dreamerv3_pdg.py
# → dreamerv3_pdg_output.png  (~10 min on GPU, ~60 min on CPU)

# Step 5: Track 2 — MPPI planner (builds on Step 4)
# Both files must be in the same directory
python mppi_dreamer.py
# → mppi_dreamer_output.png  (~15 min on GPU)
```

Each step is independent — you can run Step 3 without Step 2, and Step 5
re-trains DreamerV3 internally so you do not need to run Step 4 first
(though you can load a saved checkpoint to skip retraining).

---

## Phase Roadmap

This project follows the 6-phase plan from the research document:

| Phase | Description | File | Status |
|-------|-------------|------|--------|
| 1 | Terrain environment from DEM | `mppi_dreamer.py` → `TerrainHazardMap` | ✅ synthetic DEM; swap in real lunar/Mars DEM |
| 2 | MPPI with analytic dynamics | `mppi_dreamer.py` → `_fallback_action` | ✅ fallback = analytic brake policy |
| 3 | DreamerV3 world model | `dreamerv3_pdg.py` | ✅ full RSSM + actor-critic |
| 4 | DreamerV3 + MPPI | `mppi_dreamer.py` | ✅ prior-only rollouts, terrain cost |
| 5 | Conformal prediction | `ConformalUncertaintyHook` | 🔲 stub ready, needs calibration |
| 6 | Evaluation + baselines | `evaluate_mppi()` | ✅ MPPI vs Actor; add CP baseline |

---

## Phase 5 — Adding Conformal Prediction

The conformal hook is already wired into the cost function. To activate it:

**Step 1 — Collect calibration episodes** (hold out full episodes, not transitions):

```python
from dreamerv3_pdg import MarsPDGEnv, PDGConfig
from mppi_dreamer import ConformalUncertaintyHook
import numpy as np

env = MarsPDGEnv(PDGConfig())
cal_episodes = []

for i in range(200):              # 200 held-out episodes
    obs = env.reset(seed=8000+i)
    ep = {"obs": [], "actions": [], "rewards": [], "dones": []}
    done = False
    while not done:
        action = np.random.uniform(-1, 1, 3)  # or use trained actor
        next_obs, r, done, _ = env.step(action)
        ep["obs"].append(obs); ep["actions"].append(action)
        ep["rewards"].append(r); ep["dones"].append(float(done))
        obs = next_obs
    for k in ep: ep[k] = np.array(ep[k])
    cal_episodes.append(ep)
```

**Step 2 — Calibrate:**

```python
conf_hook = ConformalUncertaintyHook()
conf_hook.calibrate(agent, cal_episodes, alpha=0.1, device=device)
# Prints: CP calibrated: q̂ = 0.0034  (α=0.1, n=18432 calibration steps)
```

**Step 3 — Evaluate with conformal penalty:**

```python
from mppi_dreamer import evaluate_mppi, MPPIConfig, PDGConfig

eval_cp = evaluate_mppi(
    agent, terrain,
    MPPIConfig(K=512, H=20, w_conf=1.0),   # enable CP penalty
    PDGConfig(),
    n_episodes=50,
    use_conf=True,
    conf_hook=conf_hook,
    device=device,
)
```

**Step 4 — Compare to lightning-uq-box conformal calibrator** (more rigorous):

```python
from lightning_uq_box.post_hoc_conformalizers import SplitConformalClassification
# Adapt to regression: use split conformal quantile on prediction residuals
# See lightning-uq-box docs: https://lightning-uq-box.readthedocs.io
```

---

## NASA Data Sources

| Dataset | What it contains | How to get it |
|---------|-----------------|---------------|
| AI4MARS | 35K labeled Mars surface images (soil/bedrock/sand/rock) | https://data.nasa.gov → search "AI4MARS" |
| Mars Rover Photos API | Live/historical Curiosity, Opportunity, Spirit photos | https://api.nasa.gov → free key |
| Mars Curiosity Labeled (MSL) | 6,691 images, 24 classes | https://data.nasa.gov → search "Mars Surface Image Curiosity" |
| Lunar DEM (LOLA) | Lunar Orbiter Laser Altimeter elevation grids | https://pds-geosciences.wustl.edu/missions/lro/lola.htm |
| Mars MOLA DEM | Mars global topography | https://pds-geosciences.wustl.edu/missions/mgs/mola.html |
| HiRISE DTM | High-resolution Mars terrain patches | https://www.uahirise.org/dtm/ |

For Phase 1 (lunar DEM), download a LOLA tile and load it as:

```python
import numpy as np
dem = np.load("lola_tile.npy")   # or read from .tif with rasterio
terrain = TerrainHazardMap(size=dem.shape[0], resolution=30.0, dem_array=dem)
```

---

## Key Hyperparameters Reference

### DreamerV3 (`DreamerConfig` in `dreamerv3_pdg.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `deter_dim` | 512 | GRU hidden size — bigger = more memory capacity |
| `stoch_dim` | 32 | Categorical slots — controls stochastic state richness |
| `stoch_classes` | 32 | Classes per slot |
| `imag_horizon` | 15 | Imagination rollout length — longer = better long-range planning |
| `kl_free` | 1.0 | Free nats — higher = more stochastic world model |
| `kl_balance` | 0.8 | Prior vs posterior balance — do not change unless unstable |
| `actor_ent` | 3e-4 | Entropy bonus — higher = more exploration |
| `gamma` | 0.997 | Discount — high because landing takes ~100 steps |

### MPPI (`MPPIConfig` in `mppi_dreamer.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `K` | 512 | Rollout samples — more = better plan, linear cost |
| `H` | 20 | Planning horizon — 20 steps × 0.5s = 10s lookahead |
| `temperature` | 0.05 | MPPI λ — lower = winner-takes-all weighting |
| `sigma` | 0.3 | Action noise — higher = more exploration |
| `w_hazard` | 2.0 | Terrain hazard weight |
| `w_conf` | 0.0 | Conformal penalty weight — set to 1.0 after calibration |
| `safety_threshold` | 1e6 | Fallback trigger — lower = more conservative |
| `normalize_costs` | True | Normalize before temperature — keep True |

---

## Baselines

The project compares four baselines. All are implemented or stubbed:

| Baseline | How to run | Purpose |
|----------|-----------|---------|
| B1: MPPI + analytic dynamics | Set `w_conf=0, w_hazard=0` and replace RSSM with hand-coded dynamics | Does learned model add value? |
| B2: DreamerV3 Actor only | `agent.act()` without MPPI — already in `evaluate_mppi()` | Does online planning help? |
| B3: Dreamer + MPPI, no CP | Default `evaluate_mppi(use_conf=False)` | Main comparison point |
| B4: Dreamer + MPPI + CP | `evaluate_mppi(use_conf=True, conf_hook=calibrated_hook)` | Main contribution |

---

## Troubleshooting

**`ModuleNotFoundError: dreamerv3_pdg`**
All files must be in the same directory when running `mppi_dreamer.py`.

```bash
cd mars_nav/
python mppi_dreamer.py
```

**Training is very slow on CPU**
DreamerV3 is GPU-dependent for reasonable speed. On CPU, reduce:

```python
DreamerConfig(deter_dim=128, stoch_dim=16, stoch_classes=16, hidden_dim=256)
train_dreamerv3(n_env_steps=10_000, train_every=10)
```

**MPPI produces mostly fallback actions**
The world model needs more training. Increase `n_env_steps` or reduce
`safety_threshold` gradually. Also check that `normalize_costs=True`.

**`lightning-uq-box` import error**

```bash
pip install lightning-uq-box
# If still failing:
pip install lightning torch torchvision
pip install lightning-uq-box --no-deps
```

**NASA API returns no photos**
Some sols have no NAVCAM images. Try different cameras or sols:

```python
fetch_nasa_rover_image(api_key="YOUR_KEY", sol=500,  camera="MAST")
fetch_nasa_rover_image(api_key="YOUR_KEY", sol=2000, camera="FHAZ")
```

---

## References

- Hafner et al. (2025). *Mastering diverse control tasks through world models.* Nature.
  → github.com/danijar/dreamerv3
- Assran et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.*
  → github.com/facebookresearch/vjepa2
- Swan et al. (2021). *AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars.* CVPR Workshops.
  → data.nasa.gov
- Hansen et al. (2024). *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR.
  → github.com/nicklashansen/tdmpc2
- Williams et al. (2017). *Information Theoretic MPC for Model-Based Reinforcement Learning.* ICRA.
- Lehmann et al. (2025). *Lightning UQ Box: Uncertainty Quantification for Neural Networks.* JMLR.
  → github.com/lightning-uq-box/lightning-uq-box

---

## One-Sentence Summary

This project builds a hazard-aware planetary landing framework using NASA terrain data,
where DreamerV3 learns the landing world model, MPPI performs online planning over
imagined latent rollouts, and conformal prediction calibrates model uncertainty to
improve safety and robustness under terrain shift.
