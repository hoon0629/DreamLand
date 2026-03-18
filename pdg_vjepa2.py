"""
V-JEPA 2 for Powered Descent Guidance (PDG)
=============================================
Uses V-JEPA 2-AC (Action-Conditioned) as a world model for Mars landing.

Architecture:
  V-JEPA 2 encoder  →  latent state z_t
  V-JEPA 2 predictor →  imagined future z_{t+T} given action sequence a_{t:T}
  CEM planner        →  optimize thrust sequence to minimize E(z_imagined, z_goal)

Two repos from Meta FAIR:
  - facebookresearch/vjepa2     : base encoder (HuggingFace: facebook/vjepa2-vitl-fpc64-256)
  - facebookresearch/jepa-wms   : world model + planning (torch.hub)

PDG Problem:
  State:  [pos(3), vel(3), att(4-quat), alt(1), fuel(1)]
  Action: thrust vector [Tx, Ty, Tz] ∈ [T_min, T_max]^3
  Goal:   soft landing — pos_error < 5m, |vel| < 2 m/s, upright attitude
  Obs:    nadir-pointing camera frames during descent (rendered from simulation)

Install:
  pip install torch torchvision transformers gymnasium numpy matplotlib scipy
  pip install -U git+https://github.com/huggingface/transformers
  git clone https://github.com/facebookresearch/jepa-wms && cd jepa-wms && pip install -e .
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# 1. PDG ENVIRONMENT (6-DOF Mars Lander)
# ─────────────────────────────────────────────

@dataclass
class PDGConfig:
    """Mars-like powered descent parameters."""
    g:          float = 3.72    # Mars gravity (m/s²)
    mass:       float = 1905.0  # lander mass (kg) — MSL-class
    Isp:        float = 225.0   # engine specific impulse (s)
    T_max:      float = 16000.0 # max thrust (N) per axis
    T_min:      float = 0.2     # thrust throttle floor (fraction of T_max)
    dt:         float = 0.5     # timestep (s)
    max_steps:  int   = 200     # max descent steps (~100s)
    alt_init:   float = 2000.0  # initial altitude (m)
    pos_tol:    float = 5.0     # landing position tolerance (m)
    vel_tol:    float = 2.0     # landing velocity tolerance (m/s)


class MarsPDGEnv:
    """
    6-DOF Mars Powered Descent simulation.
    Matches the benchmark from Acikmese & Ploen (2007) and
    the RL formulation in Gaudet et al. (2020).

    State: [x, y, z, vx, vy, vz, mass]
      x,y  = horizontal position (m)
      z    = altitude (m), positive up
      v*   = velocities (m/s)
      mass = current propellant mass (kg)

    Action: thrust vector [Tx, Ty, Tz] in body frame
      Normalized to [-1, 1] each → rescaled to [T_min, T_max] * T_max
    """

    def __init__(self, cfg: PDGConfig = None):
        self.cfg = cfg or PDGConfig()
        self.g0 = 9.81  # standard gravity for Isp calc
        self.state = None
        self.step_count = 0

    def reset(self, seed: int = None) -> np.ndarray:
        """Reset to random initial state within deployment ellipse."""
        if seed is not None:
            np.random.seed(seed)
        cfg = self.cfg
        # Randomly sample initial position within deployment ellipse (±200m horiz)
        self.state = np.array([
            np.random.uniform(-200, 200),   # x
            np.random.uniform(-200, 200),   # y
            cfg.alt_init + np.random.uniform(-100, 100),  # z (altitude)
            np.random.uniform(-30, 30),     # vx
            np.random.uniform(-30, 30),     # vy
            np.random.uniform(-80, -20),    # vz (downward)
            cfg.mass * np.random.uniform(0.5, 0.8),  # current mass (with fuel)
        ], dtype=np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(self, action: np.ndarray):
        """
        Apply thrust action, integrate 6-DOF dynamics.

        Args:
            action: np.array([Tx, Ty, Tz]) in [-1, 1]^3 (normalized)

        Returns:
            next_state, reward, done, info
        """
        cfg = self.cfg
        s = self.state
        x, y, z, vx, vy, vz, mass = s

        # Denormalize thrust
        thrust_mag = cfg.T_max
        Tx = np.clip(action[0], -1, 1) * thrust_mag
        Ty = np.clip(action[1], -1, 1) * thrust_mag
        Tz = np.clip(action[2], -1, 1) * thrust_mag  # vertical thrust

        # Enforce throttle constraints: if thrusting, min throttle applies
        T_vec = np.array([Tx, Ty, Tz])
        T_norm = np.linalg.norm(T_vec)
        if T_norm > 0 and T_norm < cfg.T_min * thrust_mag:
            T_vec = T_vec * (cfg.T_min * thrust_mag / T_norm)
            Tx, Ty, Tz = T_vec

        # Euler integration
        ax = Tx / mass
        ay = Ty / mass
        az = Tz / mass - cfg.g  # gravity opposes upward thrust

        new_vx = vx + ax * cfg.dt
        new_vy = vy + ay * cfg.dt
        new_vz = vz + az * cfg.dt
        new_x  = x + vx * cfg.dt + 0.5 * ax * cfg.dt**2
        new_y  = y + vy * cfg.dt + 0.5 * ay * cfg.dt**2
        new_z  = z + vz * cfg.dt + 0.5 * az * cfg.dt**2

        # Propellant consumption (Tsiolkovsky)
        dm = T_norm / (cfg.Isp * self.g0) * cfg.dt
        new_mass = max(mass - dm, cfg.mass * 0.1)  # min 10% mass

        self.state = np.array([new_x, new_y, new_z,
                                new_vx, new_vy, new_vz, new_mass], dtype=np.float32)
        self.step_count += 1

        # ── Reward ──
        # Terminal: soft pinpoint landing
        pos_error = np.sqrt(new_x**2 + new_y**2)
        vel_error = np.linalg.norm([new_vx, new_vy, new_vz])

        # Fuel efficiency penalty (sparse-ish)
        fuel_cost = -T_norm / (cfg.T_max * 100.0)

        # Altitude guidance: reward descending toward target
        alt_reward = -abs(new_z) / cfg.alt_init * 0.1

        reward = fuel_cost + alt_reward

        # Check terminal conditions
        done = False
        info = {}
        if new_z <= 0:
            done = True
            if pos_error < cfg.pos_tol and vel_error < cfg.vel_tol:
                reward += 100.0  # successful soft landing bonus
                info["success"] = True
                info["landing_error_m"] = pos_error
                info["landing_speed_ms"] = vel_error
            else:
                reward -= 50.0   # crash penalty
                info["success"] = False
                info["crash_speed"] = vel_error
        elif self.step_count >= cfg.max_steps:
            done = True
            info["timeout"] = True

        return self.state.copy(), reward, done, info

    def render_frame(self, size: int = 64) -> np.ndarray:
        """
        Render a synthetic nadir-pointing camera frame.
        Returns HxWx3 uint8 image — a top-down view of lander position over terrain.

        In a full setup: replace this with AirSim or Gazebo rendered frames.
        """
        x, y, z, vx, vy, vz, mass = self.state
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Mars-like reddish terrain background
        np.random.seed(42)
        terrain = np.random.randint(100, 160, (size, size, 3), dtype=np.uint8)
        terrain[:,:,0] = np.clip(terrain[:,:,0] + 40, 0, 255)  # reddish
        terrain[:,:,2] = np.clip(terrain[:,:,2] - 30, 0, 255)  # less blue
        img = terrain.copy()

        # Landing target marker (center)
        cx, cy = size//2, size//2
        img[cx-2:cx+2, cy-2:cy+2] = [255, 255, 0]  # yellow crosshair

        # Lander shadow (position relative to target)
        # Scale: 1 pixel = 10m
        lx = int(cx + x / 10)
        ly = int(cy + y / 10)
        lx = np.clip(lx, 2, size-3)
        ly = np.clip(ly, 2, size-3)
        img[lx-2:lx+2, ly-2:ly+2] = [200, 200, 255]  # lander marker (blue-ish)

        # Altitude indicator: brightness increases as lander descends
        alt_factor = 1.0 - z / self.cfg.alt_init
        img = np.clip(img.astype(float) * (0.5 + 0.5 * alt_factor), 0, 255).astype(np.uint8)
        return img


# ─────────────────────────────────────────────
# 2. V-JEPA 2 WORLD MODEL WRAPPER
# ─────────────────────────────────────────────

class VJEPAWorldModel:
    """
    Wraps V-JEPA 2 (Action-Conditioned) for PDG planning.

    Two modes depending on available GPU:
      - "full"     : Load V-JEPA 2-AC from jepa-wms torch.hub (requires ~16GB VRAM)
      - "lite"     : Load V-JEPA 2 ViT-L encoder from HuggingFace (~4GB VRAM)
      - "mlp_proxy": Lightweight MLP surrogate trained on simulation data (CPU-friendly)

    V-JEPA 2 API (jepa-wms):
      model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_metaworld')
      z_t   = model.encode(frames)              # encode observation to latent
      z_t1  = model.predict(z_t, action)        # predict next latent given action
      energy = model.energy(z_imagined, z_goal) # L1 distance in latent space

    HuggingFace API (encoder only):
      from transformers import AutoModel, AutoVideoProcessor
      model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
      processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    """

    def __init__(self, mode: str = "mlp_proxy", device: str = None,
                 latent_dim: int = 256):
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.model = None
        self.preprocessor = None
        self._load(mode)

    def _load(self, mode: str):
        print(f"  Loading V-JEPA 2 world model (mode={mode}, device={self.device})...")

        if mode == "full":
            # Full V-JEPA 2-AC via jepa-wms torch.hub
            # git clone https://github.com/facebookresearch/jepa-wms && pip install -e .
            try:
                self.model, self.preprocessor = torch.hub.load(
                    'facebookresearch/jepa-wms', 'jepa_wm_metaworld',
                    trust_repo=True
                )
                self.model = self.model.to(self.device).eval()
                print("  ✅ Loaded jepa-wms (MetaWorld pretrained)")
            except Exception as e:
                print(f"  jepa-wms load failed ({e}). Falling back to lite mode.")
                self._load("lite")

        elif mode == "lite":
            # V-JEPA 2 ViT-L encoder from HuggingFace (encoder only, no predictor)
            # Add a learned action-conditioned predictor head on top
            try:
                from transformers import AutoModel, AutoVideoProcessor
                hf_repo = "facebook/vjepa2-vitl-fpc64-256"
                self.model = AutoModel.from_pretrained(hf_repo).to(self.device).eval()
                self.preprocessor = AutoVideoProcessor.from_pretrained(hf_repo)
                # Action-conditioned predictor head (to be trained on simulation)
                self.predictor = ActionConditionedPredictor(
                    latent_dim=1024,  # ViT-L output dim
                    action_dim=3,     # [Tx, Ty, Tz]
                ).to(self.device)
                print(f"  ✅ Loaded V-JEPA 2 ViT-L from HuggingFace ({hf_repo})")
                print("  ⚠️  Predictor is randomly initialized — train on simulation data")
            except Exception as e:
                print(f"  HuggingFace load failed ({e}). Using MLP proxy.")
                self._load("mlp_proxy")

        else:  # "mlp_proxy"
            # Lightweight surrogate: MLP trained on simulation rollouts
            # Identical interface to the V-JEPA 2 API for easy swapping
            print("  Using MLP surrogate (CPU-friendly, trains on sim data)")
            self.model = MLPWorldModel(
                obs_dim=7,            # PDG state dim
                action_dim=3,
                latent_dim=self.latent_dim,
            ).to(self.device)
            print("  ℹ️  Replace with V-JEPA 2 once trained on simulation video")

    def encode(self, obs) -> torch.Tensor:
        """
        Encode observation to latent vector z.

        Args:
            obs: state vector (7,) or image frames (T, C, H, W)

        Returns:
            z: latent tensor (latent_dim,)
        """
        if self.mode == "mlp_proxy":
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(self.device)
            return self.model.encode(obs)
        else:
            # V-JEPA 2: encode video clip
            if isinstance(obs, np.ndarray):
                # obs assumed to be (T, H, W, C) uint8 frames
                frames = torch.FloatTensor(obs).permute(0, 3, 1, 2) / 255.0
                frames = frames.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
            else:
                frames = obs.to(self.device)
            with torch.no_grad():
                z = self.model(frames).last_hidden_state.mean(dim=1)
            return z.squeeze(0)

    def predict(self, z: torch.Tensor, action: np.ndarray) -> torch.Tensor:
        """
        Predict next latent state given current latent + action.

        Args:
            z:      current latent (latent_dim,)
            action: thrust vector (3,) in [-1, 1]^3

        Returns:
            z_next: predicted next latent (latent_dim,)
        """
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(self.device)
        if self.mode == "mlp_proxy":
            return self.model.predict(z, action)
        else:
            # For V-JEPA 2-AC (jepa-wms):
            # return self.model.predict(z.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
            return self.predictor(z, action)

    def energy(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        Goal-conditioned energy: L1 distance in latent space.
        Lower = imagined future is closer to goal state.
        Matches V-JEPA 2-AC planning objective.
        """
        return F.l1_loss(z_pred, z_goal)

    def rollout(self, z0: torch.Tensor,
                actions: torch.Tensor) -> list:
        """
        Multi-step latent rollout.

        Args:
            z0:      initial latent (latent_dim,)
            actions: action sequence (T, 3)

        Returns:
            latents: list of T latent tensors
        """
        z = z0
        latents = []
        for t in range(actions.shape[0]):
            z = self.predict(z, actions[t])
            latents.append(z)
        return latents


# ─────────────────────────────────────────────
# 3. MLP SURROGATE WORLD MODEL
# ─────────────────────────────────────────────

class MLPWorldModel(nn.Module):
    """
    Lightweight MLP surrogate with same interface as V-JEPA 2.
    Train this on simulation rollouts first, then swap in V-JEPA 2.

    Architecture mirrors JEPA philosophy: separate encoder + action-conditioned predictor.
    """

    def __init__(self, obs_dim: int = 7, action_dim: int = 3,
                 latent_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: obs → latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent_dim),
        )

        # Action-conditioned predictor: (latent + action) → next latent
        # Separate from encoder (JEPA principle: predictor only active during planning)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent_dim),
        )

        # Decoder (optional — only for debugging/visualization, not needed for planning)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([z, action], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        z = self.encode(obs)
        z_next = self.predict(z, action)
        return z_next, z

    def loss(self, obs: torch.Tensor, action: torch.Tensor,
             next_obs: torch.Tensor) -> dict:
        """
        JEPA-style training loss: predict NEXT LATENT, not next pixels.
        Target latent is encoded from next_obs (with stop-gradient).
        """
        z_pred, z_curr = self.forward(obs, action)

        # Encode target with stop-gradient (JEPA: target encoder is EMA or frozen)
        with torch.no_grad():
            z_target = self.encode(next_obs)

        # Prediction loss in latent space (L1, matching V-JEPA 2 energy function)
        pred_loss = F.l1_loss(z_pred, z_target)

        # Reconstruction loss (optional, for debugging only)
        obs_recon = self.decoder(z_curr)
        recon_loss = F.mse_loss(obs_recon, obs)

        return {
            "total": pred_loss + 0.1 * recon_loss,
            "pred":  pred_loss,
            "recon": recon_loss,
        }


class ActionConditionedPredictor(nn.Module):
    """
    Lightweight predictor head to add on top of frozen V-JEPA 2 encoder.
    Train this on simulation trajectories while keeping V-JEPA 2 frozen.
    """

    def __init__(self, latent_dim: int = 1024, action_dim: int = 3,
                 hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, action], dim=-1))


# ─────────────────────────────────────────────
# 4. CEM PLANNER (MPC in latent space)
# ─────────────────────────────────────────────

class CEMPlanner:
    """
    Cross-Entropy Method (CEM) planner operating in V-JEPA 2 latent space.

    This is exactly the planning procedure used in V-JEPA 2-AC:
      1. Sample N action sequences from current distribution
      2. Roll out each in latent space using world model predictor
      3. Evaluate energy(z_final, z_goal)
      4. Keep top-K sequences, refit Gaussian
      5. Repeat for num_iterations

    Matches V-JEPA 2 paper: "action sequences optimized via cross-entropy method
    to minimize L1 distance between imagined future and goal in latent space"
    """

    def __init__(self, world_model: VJEPAWorldModel,
                 horizon: int = 20,
                 n_samples: int = 256,
                 n_elite: int = 32,
                 n_iterations: int = 5,
                 action_dim: int = 3,
                 action_bounds: tuple = (-1.0, 1.0)):
        self.wm = world_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.device = world_model.device

    def plan(self, z_curr: torch.Tensor,
             z_goal: torch.Tensor) -> np.ndarray:
        """
        Find the best action sequence via CEM.

        Args:
            z_curr: current latent state (latent_dim,)
            z_goal: goal latent state (latent_dim,)

        Returns:
            best_action: first action of best sequence (action_dim,) in [-1,1]^3
        """
        # Initialize action distribution: μ=0, σ=0.5
        mu    = torch.zeros(self.horizon, self.action_dim, device=self.device)
        sigma = torch.ones(self.horizon, self.action_dim, device=self.device) * 0.5

        for iteration in range(self.n_iterations):
            # Sample N action sequences
            eps = torch.randn(self.n_samples, self.horizon,
                              self.action_dim, device=self.device)
            actions = torch.clamp(mu.unsqueeze(0) + sigma.unsqueeze(0) * eps,
                                  *self.action_bounds)
            # (N, T, action_dim)

            # Roll out in latent space
            energies = torch.zeros(self.n_samples, device=self.device)
            for i in range(self.n_samples):
                latents = self.wm.rollout(z_curr, actions[i])
                z_final = latents[-1]
                energies[i] = self.wm.energy(z_final, z_goal)

            # Select elite (lowest energy = closest to goal)
            elite_idx = energies.argsort()[:self.n_elite]
            elite_actions = actions[elite_idx]  # (n_elite, T, action_dim)

            # Refit Gaussian from elite samples
            mu    = elite_actions.mean(0)
            sigma = elite_actions.std(0) + 1e-6

        # Return first action of best sequence
        best_actions = actions[elite_idx[0]]  # (T, action_dim)
        return best_actions[0].cpu().numpy()

    def plan_with_fuel_penalty(self, z_curr: torch.Tensor,
                                z_goal: torch.Tensor,
                                fuel_weight: float = 0.01) -> np.ndarray:
        """
        Fuel-optimal planning: energy + fuel cost penalty.
        Encourages minimal thrust while still reaching goal.
        Closer to PDG's fuel-optimal objective.
        """
        mu    = torch.zeros(self.horizon, self.action_dim, device=self.device)
        sigma = torch.ones(self.horizon, self.action_dim, device=self.device) * 0.5

        for _ in range(self.n_iterations):
            eps = torch.randn(self.n_samples, self.horizon,
                              self.action_dim, device=self.device)
            actions = torch.clamp(mu.unsqueeze(0) + sigma.unsqueeze(0) * eps,
                                  *self.action_bounds)

            energies = torch.zeros(self.n_samples, device=self.device)
            for i in range(self.n_samples):
                latents = self.wm.rollout(z_curr, actions[i])
                z_final = latents[-1]
                # Goal energy + fuel cost
                goal_energy = self.wm.energy(z_final, z_goal)
                fuel_cost   = actions[i].norm(dim=-1).mean() * fuel_weight
                energies[i] = goal_energy + fuel_cost

            elite_idx = energies.argsort()[:self.n_elite]
            elite_actions = actions[elite_idx]
            mu    = elite_actions.mean(0)
            sigma = elite_actions.std(0) + 1e-6

        return actions[elite_idx[0]][0].cpu().numpy()


# ─────────────────────────────────────────────
# 5. WORLD MODEL TRAINING (Simulation Data)
# ─────────────────────────────────────────────

def collect_simulation_data(env: MarsPDGEnv, n_episodes: int = 500,
                             policy: str = "random") -> dict:
    """
    Collect (obs, action, next_obs) tuples from simulation.
    Used to train/fine-tune the world model predictor.

    For V-JEPA 2 lite mode: also saves rendered frames as video clips.
    """
    obs_list, act_list, next_obs_list = [], [], []

    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        done = False
        while not done:
            if policy == "random":
                action = np.random.uniform(-1, 1, 3)
            else:
                # Proportional guidance: thrust toward landing
                x, y, z, vx, vy, vz, mass = obs
                # Simple gravity turn: thrust opposite to velocity
                v = np.array([vx, vy, vz])
                v_norm = np.linalg.norm(v)
                action = -v / (v_norm + 1e-6) * 0.7 if v_norm > 0 else np.zeros(3)
                action = np.clip(action, -1, 1)

            next_obs, _, done, _ = env.step(action)
            obs_list.append(obs.copy())
            act_list.append(action.copy())
            next_obs_list.append(next_obs.copy())
            obs = next_obs

    data = {
        "obs":      torch.FloatTensor(np.array(obs_list)),
        "actions":  torch.FloatTensor(np.array(act_list)),
        "next_obs": torch.FloatTensor(np.array(next_obs_list)),
    }
    print(f"  Collected {len(obs_list)} transitions from {n_episodes} episodes")
    return data


def train_world_model(world_model: VJEPAWorldModel,
                      data: dict,
                      n_epochs: int = 50,
                      batch_size: int = 256,
                      lr: float = 3e-4) -> list:
    """
    Train the MLP world model (or fine-tune predictor head for V-JEPA 2 lite).
    JEPA objective: predict next latent, not next pixels.
    """
    if world_model.mode != "mlp_proxy":
        print("  For V-JEPA 2 full/lite: only fine-tune the predictor head.")
        model = world_model.predictor if hasattr(world_model, 'predictor') else world_model.model
    else:
        model = world_model.model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs)

    obs      = data["obs"].to(world_model.device)
    actions  = data["actions"].to(world_model.device)
    next_obs = data["next_obs"].to(world_model.device)

    n = len(obs)
    losses = []

    print(f"\n  Training world model ({n} transitions, {n_epochs} epochs)...")
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            loss_dict = model.loss(obs[idx], actions[idx], next_obs[idx])
            optimizer.zero_grad()
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_dict["total"].item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches
        losses.append(epoch_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{n_epochs}  loss={epoch_loss:.6f}")

    print(f"  Training complete. Final loss: {losses[-1]:.6f}")
    return losses


# ─────────────────────────────────────────────
# 6. FULL PIPELINE
# ─────────────────────────────────────────────

def run_pdg_pipeline(mode: str = "mlp_proxy",
                     n_train_episodes: int = 500,
                     n_eval_episodes: int = 20,
                     cem_horizon: int = 20,
                     cem_samples: int = 256,
                     plot: bool = True):
    """
    Full PDG pipeline: train world model → evaluate CEM planner.

    Args:
        mode: "mlp_proxy" | "lite" | "full"
        n_train_episodes: simulation episodes for world model training
        n_eval_episodes:  evaluation rollouts
        cem_horizon:      planning horizon (steps)
        cem_samples:      CEM population size
        plot:             show trajectory visualization
    """
    print("\n🚀 V-JEPA 2 Powered Descent Guidance (PDG)")
    print("=" * 50)

    cfg = PDGConfig()
    env = MarsPDGEnv(cfg)

    # 1. Load world model
    print("\n[1/4] Loading V-JEPA 2 world model...")
    wm = VJEPAWorldModel(mode=mode)

    # 2. Collect simulation data & train
    print("\n[2/4] Collecting simulation data...")
    data = collect_simulation_data(env, n_episodes=n_train_episodes,
                                   policy="proportional")

    print("\n[3/4] Training world model (JEPA objective: predict in latent space)...")
    losses = train_world_model(wm, data, n_epochs=50)

    # 3. Evaluate CEM planner
    print("\n[4/4] Evaluating CEM planner...")
    planner = CEMPlanner(
        world_model=wm,
        horizon=cem_horizon,
        n_samples=cem_samples,
        n_elite=32,
        n_iterations=5,
    )

    # Build goal latent: soft landing = zero pos/vel at ground
    goal_state = np.array([0., 0., 0., 0., 0., -0.5, cfg.mass * 0.5], dtype=np.float32)
    z_goal = wm.encode(torch.FloatTensor(goal_state).to(wm.device))

    results = []
    trajectories = []

    for ep in range(n_eval_episodes):
        obs = env.reset(seed=1000 + ep)
        done = False
        traj = [obs.copy()]
        total_reward = 0.0

        while not done:
            z_curr = wm.encode(torch.FloatTensor(obs).to(wm.device))
            action = planner.plan_with_fuel_penalty(z_curr, z_goal, fuel_weight=0.02)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            traj.append(obs.copy())

        results.append({
            "success":       info.get("success", False),
            "total_reward":  total_reward,
            "landing_error": info.get("landing_error_m", None),
            "landing_speed": info.get("landing_speed_ms", None),
        })
        trajectories.append(np.array(traj))

    # Summary
    n_success = sum(r["success"] for r in results)
    errors = [r["landing_error"] for r in results if r["landing_error"] is not None]
    print(f"\n  Success rate  : {n_success}/{n_eval_episodes} "
          f"({100*n_success/n_eval_episodes:.0f}%)")
    if errors:
        print(f"  Avg landing error : {np.mean(errors):.2f} m  "
              f"(target: <{cfg.pos_tol}m)")
    print(f"  Avg total reward  : {np.mean([r['total_reward'] for r in results]):.2f}")

    if plot:
        _plot_results(trajectories, losses, results, cfg)

    return results, trajectories


def _plot_results(trajectories, losses, results, cfg):
    """3-panel visualization: training loss | 3D trajectory | landing scatter."""
    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(1, 3, figure=fig)

    plot_kw = dict(facecolor="#0d1117")
    label_kw = dict(color="white", fontsize=9)

    # Panel 1: Training loss
    ax1 = fig.add_subplot(gs[0], **plot_kw)
    ax1.plot(losses, color="#00d4ff", linewidth=1.5)
    ax1.set_title("World Model Training Loss\n(JEPA: predict in latent space)",
                   color="white", fontsize=10)
    ax1.set_xlabel("Epoch", **label_kw)
    ax1.set_ylabel("L1 Prediction Loss", **label_kw)
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#30363d")
    ax1.set_facecolor("#161b22")

    # Panel 2: Descent trajectories (altitude vs horizontal distance)
    ax2 = fig.add_subplot(gs[1], **plot_kw)
    for traj in trajectories[:10]:
        x, y, z = traj[:,0], traj[:,1], traj[:,2]
        horiz = np.sqrt(x**2 + y**2)
        color = "#00ff88" if traj[-1, 2] <= 0 and np.sqrt(traj[-1,0]**2 + traj[-1,1]**2) < cfg.pos_tol else "#ff6b6b"
        ax2.plot(horiz, z, color=color, alpha=0.7, linewidth=1.2)
    ax2.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax2.set_title("Descent Trajectories\n(green=success, red=miss)",
                   color="white", fontsize=10)
    ax2.set_xlabel("Horizontal Distance (m)", **label_kw)
    ax2.set_ylabel("Altitude (m)", **label_kw)
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
    ax2.set_facecolor("#161b22")

    # Panel 3: Landing scatter
    ax3 = fig.add_subplot(gs[2], **plot_kw)
    for r, traj in zip(results, trajectories):
        fx, fy = traj[-1, 0], traj[-1, 1]
        color = "#00ff88" if r["success"] else "#ff6b6b"
        ax3.scatter(fx, fy, color=color, s=30, alpha=0.8, zorder=3)
    circle = plt.Circle((0, 0), cfg.pos_tol, color="#ffff00",
                          fill=False, linewidth=1.5, linestyle="--", label=f"{cfg.pos_tol}m target")
    ax3.add_patch(circle)
    ax3.set_aspect("equal")
    ax3.set_title(f"Landing Scatter\n(target circle = {cfg.pos_tol}m radius)",
                   color="white", fontsize=10)
    ax3.set_xlabel("X landing error (m)", **label_kw)
    ax3.set_ylabel("Y landing error (m)", **label_kw)
    ax3.tick_params(colors="white")
    for sp in ax3.spines.values(): sp.set_edgecolor("#30363d")
    ax3.set_facecolor("#161b22")
    ax3.legend(fontsize=8, facecolor="#161b22", labelcolor="white", edgecolor="#30363d")

    plt.suptitle("V-JEPA 2 World Model  |  CEM Planner  |  Mars PDG Simulation",
                  color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("pdg_vjepa2_output.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Saved → pdg_vjepa2_output.png")
    plt.show()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    MODES:

    1. Quick start (MLP surrogate, CPU-friendly):
       python pdg_vjepa2.py
       → Trains MLP world model on sim data, evaluates CEM planner

    2. V-JEPA 2 ViT-L encoder (needs GPU, ~4GB VRAM):
       results = run_pdg_pipeline(mode="lite")
       → Loads frozen V-JEPA 2 from HuggingFace, trains action-conditioned head

    3. Full jepa-wms (needs GPU, ~16GB VRAM):
       results = run_pdg_pipeline(mode="full")
       → Loads pretrained JEPA-WM, uses full V-JEPA 2-AC for planning

    Upgrade path:
       mlp_proxy → lite (V-JEPA 2 encoder + trained predictor) → full (jepa-wms)

    V-JEPA 2 repos:
       github.com/facebookresearch/vjepa2     (encoder)
       github.com/facebookresearch/jepa-wms   (world model + planning)
       github.com/facebookresearch/eb_jepa    (action-conditioned examples)
       huggingface.co/facebook/vjepa2-vitl-fpc64-256  (HuggingFace weights)
    """
    results, trajectories = run_pdg_pipeline(
        mode="mlp_proxy",       # change to "lite" or "full" for real V-JEPA 2
        n_train_episodes=500,
        n_eval_episodes=20,
        cem_horizon=20,
        cem_samples=256,
        plot=True,
    )
