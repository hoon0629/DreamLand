"""
MPPI Planner over DreamerV3 Latent World Model
================================================
Model Predictive Path Integral (MPPI) using the DreamerV3 RSSM
as the rollout dynamics model for Mars hazard-aware landing.

Key interface design (from project review):
  - Posterior is called ONCE per real timestep to encode obs → (h_t, z_t)
  - MPPI inner loop uses PRIOR ONLY for all K×H imagined rollouts
  - This is correct: MPPI doesn't have K×H real env observations
  - Temperature λ is kept separate from conformal penalty scale

Cost function:
  J_k = Σ_{t=0}^{H-1} [c_task + λ1·c_hazard + λ2·c_control + λ3·c_conf] + c_terminal

  c_task:    distance to landing target + velocity penalty
  c_hazard:  terrain slope/roughness at predicted horizontal position
  c_control: L2 norm of thrust (fuel efficiency)
  c_conf:    conformal uncertainty penalty (placeholder → filled by CP module)

MPPI update rule:
  w_k  = exp(-J_k / λ) / Σ_j exp(-J_j / λ)
  ã_t  = Σ_k w_k · a_{k,t}             (weighted mean)
  σ_t² kept fixed or adapted per iter

Fallback policy (when all K rollouts exceed safety threshold):
  → Brake: thrust upward to reduce descent rate
  → Prevents unconstrained crash when model is maximally uncertain

Install:
  pip install torch numpy matplotlib scipy
  (dreamerv3_pdg.py must be in the same directory)

Reference:
  Williams et al. (2017) "Information Theoretic MPC for Model-Based RL"
  Hafner et al. (2025) "Mastering Diverse Control Tasks through World Models"
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import deque
import time
import warnings
warnings.filterwarnings("ignore")

# Import DreamerV3 components from our existing module
from dreamerv3_pdg import (
    DreamerV3, DreamerConfig, PDGConfig, MarsPDGEnv,
    symlog, symexp, mlp,
)


# ─────────────────────────────────────────────
# MPPI CONFIG
# ─────────────────────────────────────────────

@dataclass
class MPPIConfig:
    # Sampling
    K:             int   = 512    # number of rollout samples
    H:             int   = 20     # planning horizon (steps)
    action_dim:    int   = 3      # [Tx, Ty, Tz] ∈ [-1, 1]^3

    # MPPI temperature
    # Controls how sharply good rollouts are upweighted.
    # Lower λ → winner-takes-all; higher λ → more averaging.
    # NOTE: keep λ and cost scales consistent — see normalize_costs flag.
    temperature:   float = 0.05

    # Noise schedule
    # Action perturbation σ per dimension. Higher → more exploration.
    sigma:         float = 0.3
    # Optionally per-dimension: [σ_x, σ_y, σ_z]
    sigma_vec:     Optional[list] = None

    # Cost weights
    w_task:        float = 1.0    # goal-reaching cost
    w_vel:         float = 0.5    # velocity penalty (fuel for braking)
    w_control:     float = 0.05   # thrust magnitude (fuel cost)
    w_hazard:      float = 2.0    # terrain hazard penalty
    w_conf:        float = 0.0    # conformal uncertainty (0 = disabled until CP added)

    # Terminal cost multiplier
    w_terminal:    float = 5.0

    # Safety: max allowed cost per rollout (for fallback trigger)
    # If ALL K rollouts exceed this, fallback policy activates.
    # Set to np.inf to disable fallback.
    safety_threshold: float = 1e6

    # Normalize costs to [0,1] before computing weights.
    # Recommended: True — prevents temperature sensitivity to cost scale.
    normalize_costs:   bool = True

    # Action warm-starting: shift previous solution forward each step.
    warm_start:    bool = True

    # Number of MPPI iterations per control step (>1 → receding-horizon refinement).
    n_iter:        int  = 1

    # Clamp final actions to valid range.
    action_low:    float = -1.0
    action_high:   float =  1.0


# ─────────────────────────────────────────────
# TERRAIN HAZARD MODEL
# ─────────────────────────────────────────────

class TerrainHazardMap:
    """
    Terrain-aware hazard scoring from a DEM (elevation grid).

    Derives per-cell hazard score from:
      slope     = ||∇DEM||   (gradient magnitude)
      roughness = std(DEM in local window)
      hazard    = clip(w_slope*slope + w_rough*roughness, 0, 1)

    For Phase 1 (simplified): uses a synthetic hazard field.
    For Phase 2: replace __init__ with real lunar/Mars DEM loading.

    Usage:
      hazard_map = TerrainHazardMap(size=200, resolution=5.0)  # 200×200 grid, 5m/cell
      h = hazard_map.query(x, y)  # scalar hazard at (x,y) meters from origin
    """

    def __init__(self, size: int = 200, resolution: float = 5.0,
                 dem_array: Optional[np.ndarray] = None, seed: int = 42):
        """
        Args:
            size:        grid side length in cells
            resolution:  meters per cell
            dem_array:   (size, size) elevation array in meters.
                         If None, generates synthetic Mars-like terrain.
            seed:        random seed for synthetic terrain
        """
        self.size       = size
        self.resolution = resolution  # m/cell
        self.extent     = size * resolution / 2  # half-width in meters

        if dem_array is not None:
            self.dem = dem_array.astype(np.float32)
        else:
            self.dem = self._synthetic_dem(size, seed)

        # Derive hazard map
        self.slope      = self._compute_slope()
        self.roughness  = self._compute_roughness(window=5)
        self.hazard_map = self._compute_hazard(w_slope=0.7, w_rough=0.3)

    # ── DEM Derivatives ──────────────────────────────────────

    def _synthetic_dem(self, size: int, seed: int) -> np.ndarray:
        """
        Generate a synthetic Mars-like DEM with craters and rocky terrain.
        Replace with: dem = np.load("lunar_dem_tile.npy") for real data.
        """
        np.random.seed(seed)
        # Smooth base terrain
        from scipy.ndimage import gaussian_filter
        base = np.random.randn(size, size) * 10.0
        dem  = gaussian_filter(base, sigma=8)

        # Add craters (circular depressions)
        n_craters = 12
        for _ in range(n_craters):
            cx   = np.random.randint(10, size-10)
            cy   = np.random.randint(10, size-10)
            r    = np.random.randint(5, 20)
            depth= np.random.uniform(5, 25)
            Y, X = np.ogrid[:size, :size]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            dem[dist < r] -= depth * (1 - dist[dist < r] / r)

        # Add boulders / rocky patches
        n_boulders = 30
        for _ in range(n_boulders):
            bx = np.random.randint(0, size)
            by = np.random.randint(0, size)
            bh = np.random.uniform(1, 5)
            br = np.random.randint(1, 4)
            Y, X = np.ogrid[:size, :size]
            mask = np.sqrt((X - bx)**2 + (Y - by)**2) < br
            dem[mask] += bh

        return dem.astype(np.float32)

    def _compute_slope(self) -> np.ndarray:
        """Slope = ||∇DEM|| / resolution, normalized to [0,1]."""
        gy, gx = np.gradient(self.dem, self.resolution)
        slope  = np.sqrt(gx**2 + gy**2)
        return (slope / (slope.max() + 1e-8)).astype(np.float32)

    def _compute_roughness(self, window: int = 5) -> np.ndarray:
        """Roughness = local std dev of elevation in (window×window) patch."""
        from scipy.ndimage import generic_filter
        roughness = generic_filter(self.dem, np.std, size=window)
        return (roughness / (roughness.max() + 1e-8)).astype(np.float32)

    def _compute_hazard(self, w_slope: float = 0.7,
                         w_rough: float = 0.3) -> np.ndarray:
        """Combined hazard score ∈ [0, 1]. 0=safe, 1=dangerous."""
        hazard = np.clip(w_slope * self.slope + w_rough * self.roughness, 0, 1)
        return hazard.astype(np.float32)

    # ── Query ─────────────────────────────────────────────────

    def _world_to_grid(self, x: np.ndarray, y: np.ndarray):
        """Convert world coordinates (meters) to grid indices."""
        col = np.clip(((x + self.extent) / self.resolution).astype(int),
                      0, self.size - 1)
        row = np.clip(((y + self.extent) / self.resolution).astype(int),
                      0, self.size - 1)
        return row, col

    def query(self, x, y) -> np.ndarray:
        """
        Query hazard score at world position (x, y) in meters.
        Supports scalar or array inputs.
        Returns float or array in [0, 1].
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float32))
        y = np.atleast_1d(np.asarray(y, dtype=np.float32))
        row, col = self._world_to_grid(x, y)
        return self.hazard_map[row, col]

    def query_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Batched torch query for MPPI rollouts."""
        x_np = x.cpu().numpy().astype(np.float32)
        y_np = y.cpu().numpy().astype(np.float32)
        h    = self.query(x_np, y_np)
        return torch.FloatTensor(h).to(x.device)

    def plot(self, ax=None, title="Terrain Hazard Map"):
        """Visualize the hazard map."""
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 5))
        extent = [-self.extent, self.extent, -self.extent, self.extent]
        im = ax.imshow(self.hazard_map, cmap="RdYlGn_r",
                       vmin=0, vmax=1, extent=extent, origin="lower")
        plt.colorbar(im, ax=ax, label="Hazard (0=safe, 1=dangerous)")
        ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        return ax


# ─────────────────────────────────────────────
# MPPI COST FUNCTION
# ─────────────────────────────────────────────

class MPPICostFunction:
    """
    Computes per-step and terminal costs for MPPI rollouts.

    All costs are computed in latent-decoded state space.
    The decoder maps [h_t, z_t] → predicted obs [x, y, z, vx, vy, vz, mass].

    Cost terms:
      c_task:    ||[x,y]||² / norm  +  ||vz||² / norm   (reach target, slow down)
      c_vel:     ||[vx,vy,vz]||² / norm                  (penalize high velocity)
      c_control: ||action||² / dim                        (minimize thrust = save fuel)
      c_hazard:  terrain_hazard(x_pred, y_pred)           (terrain-aware)
      c_conf:    conformal_penalty(latent_state)          (placeholder for CP module)
      c_terminal: large penalty for crash / large pos/vel error at t=H
    """

    def __init__(self, cfg: MPPIConfig,
                 terrain: Optional[TerrainHazardMap] = None,
                 pdg_cfg: Optional[PDGConfig] = None,
                 conf_fn: Optional[Callable] = None):
        """
        Args:
            cfg:      MPPI config
            terrain:  TerrainHazardMap instance (or None → no hazard cost)
            pdg_cfg:  PDG environment config (for normalization)
            conf_fn:  Callable(h, z) → scalar uncertainty [0,1].
                      Plug in CP module here. None → c_conf = 0.
        """
        self.cfg     = cfg
        self.terrain = terrain
        self.pdg_cfg = pdg_cfg or PDGConfig()
        self.conf_fn = conf_fn  # ← CP hook: fill in Phase 5

        # Normalization constants (from PDGConfig)
        c = self.pdg_cfg
        self.pos_norm  = c.alt_init        # ~2000m
        self.vel_norm  = 100.0             # ~100 m/s max speed
        self.alt_norm  = c.alt_init

    def step_cost(self, decoded_obs: torch.Tensor,
                  action: torch.Tensor,
                  h: Optional[torch.Tensor] = None,
                  z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Per-step cost for K rollouts at one timestep.

        Args:
            decoded_obs: (K, obs_dim) predicted state via decoder
            action:      (K, action_dim) applied action
            h, z:        latent state for conformal query (optional)

        Returns:
            cost: (K,) per-sample cost
        """
        cfg = self.cfg
        x   = decoded_obs[:, 0];  y  = decoded_obs[:, 1]
        z_  = decoded_obs[:, 2]   # altitude (not latent z)
        vx  = decoded_obs[:, 3];  vy = decoded_obs[:, 4]; vz = decoded_obs[:, 5]

        # c_task: horizontal distance to target (origin) + altitude penalty
        pos_err  = (x**2 + y**2) / self.pos_norm**2
        alt_pen  = torch.clamp(z_, min=0) / self.alt_norm  # penalize remaining alt
        c_task   = pos_err + 0.1 * alt_pen

        # c_vel: total speed (want slow landing)
        speed    = (vx**2 + vy**2 + vz**2) / self.vel_norm**2
        c_vel    = speed

        # c_control: thrust magnitude
        c_ctrl   = (action**2).mean(-1)

        # c_hazard: terrain hazard at predicted landing footprint
        if self.terrain is not None and cfg.w_hazard > 0:
            c_haz = self.terrain.query_torch(x, y)
        else:
            c_haz = torch.zeros_like(x)

        # c_conf: conformal uncertainty penalty (placeholder)
        # Replace with: c_conf = self.conf_fn(h, z) when CP is added
        if self.conf_fn is not None and h is not None and cfg.w_conf > 0:
            c_conf = self.conf_fn(h, z)
        else:
            c_conf = torch.zeros_like(x)

        cost = (cfg.w_task    * c_task   +
                cfg.w_vel     * c_vel    +
                cfg.w_control * c_ctrl   +
                cfg.w_hazard  * c_haz    +
                cfg.w_conf    * c_conf)

        return cost

    def terminal_cost(self, decoded_obs: torch.Tensor) -> torch.Tensor:
        """
        Terminal cost at end of H-step rollout.
        Heavily penalizes large final position error and velocity.

        Args:
            decoded_obs: (K, obs_dim)
        Returns:
            cost: (K,)
        """
        cfg = self.cfg
        x   = decoded_obs[:, 0];  y  = decoded_obs[:, 1]
        z_  = decoded_obs[:, 2]
        vx  = decoded_obs[:, 3];  vy = decoded_obs[:, 4]; vz = decoded_obs[:, 5]

        pos_err  = torch.sqrt(x**2 + y**2) / self.pos_norm
        vel_err  = torch.sqrt(vx**2 + vy**2 + vz**2) / self.vel_norm
        alt_err  = torch.clamp(z_, min=0) / self.alt_norm

        # Penalty for being below ground (crash prediction)
        crash_pen = torch.clamp(-z_, min=0) * 10.0

        cost = cfg.w_terminal * (pos_err + vel_err + alt_err + crash_pen)
        return cost


# ─────────────────────────────────────────────
# MPPI PLANNER
# ─────────────────────────────────────────────

class MPPIPlanner:
    """
    Model Predictive Path Integral planner using DreamerV3 RSSM.

    MPPI algorithm (Williams et al. 2017):
    ─────────────────────────────────────
    Given current state s_t and nominal action sequence U = {u_0,...,u_{H-1}}:

    1. Sample K perturbations: ε_{k,t} ~ N(0, Σ)
       → V_{k,t} = u_t + ε_{k,t}   (sampled action sequence for rollout k)

    2. Roll out K sequences in world model (PRIOR ONLY):
       For k=1..K, t=0..H-1:
         [h_{k,t+1}, z_{k,t+1}] = RSSM.step_prior(h_{k,t}, z_{k,t}, V_{k,t})
         obs_{k,t}               = Decoder([h_{k,t}, z_{k,t}])
         J_k += cost_fn(obs_{k,t}, V_{k,t})
       J_k += terminal_cost(obs_{k,H})

    3. Compute importance weights:
       β   = min_k J_k   (for numerical stability)
       η   = Σ_k exp(-(J_k - β) / λ)
       w_k = exp(-(J_k - β) / λ) / η

    4. Update nominal action sequence:
       u_t ← u_t + Σ_k w_k · ε_{k,t}   for t=0..H-1

    5. Execute u_0. Shift U left: u_t ← u_{t+1} for t=0..H-2, u_{H-1} ← 0.

    Key design decisions:
      - Posterior called ONCE per real step (not K times)
      - Prior used for all K×H inner rollouts
      - Costs normalized before weighting (temperature stability)
      - Fallback policy activates when all rollouts are unsafe
    """

    def __init__(self,
                 agent: DreamerV3,
                 mppi_cfg: MPPIConfig,
                 cost_fn: MPPICostFunction,
                 device: str = None):
        self.agent   = agent
        self.cfg     = mppi_cfg
        self.cost_fn = cost_fn
        self.device  = device or next(agent.parameters()).device

        dream_cfg = agent.cfg
        self.deter_dim   = dream_cfg.deter_dim
        self.stoch_flat  = dream_cfg.stoch_dim * dream_cfg.stoch_classes
        self.action_dim  = mppi_cfg.action_dim

        # Nominal action sequence U: (H, action_dim)
        self._reset_nominal()

        # Per-dim sigma vector
        if mppi_cfg.sigma_vec is not None:
            self.sigma = torch.FloatTensor(mppi_cfg.sigma_vec).to(self.device)
        else:
            self.sigma = torch.full((mppi_cfg.action_dim,),
                                    mppi_cfg.sigma, device=self.device)

        # Stats for logging
        self.last_costs   = None
        self.last_weights = None
        self.last_best_J  = None
        self.n_fallbacks  = 0

    def _reset_nominal(self):
        """Initialize or reset nominal action sequence to zeros."""
        self.U = torch.zeros(self.cfg.H, self.cfg.action_dim,
                             device=self.device)

    @torch.no_grad()
    def plan(self,
             obs: np.ndarray,
             h: torch.Tensor,
             z: torch.Tensor) -> tuple:
        """
        Run MPPI to select next action.

        Args:
            obs: current real observation (obs_dim,)
            h:   current RSSM deterministic state (1, deter_dim)
            z:   current RSSM stochastic state (1, stoch_flat)

        Returns:
            action:  (action_dim,) numpy array — action to execute
            h_new:   updated h after posterior step (1, deter_dim)
            z_new:   updated z after posterior step (1, stoch_flat)
            info:    dict with planning diagnostics
        """
        cfg = self.cfg

        # ── Step 1: Posterior update (ONCE per real timestep)
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        embed  = self.agent.encoder(obs_t)
        # Use a zero action for the posterior step (we just updated to current obs)
        a_prev = torch.zeros(1, self.action_dim, device=self.device)
        h_new, z_new, _, _ = self.agent.rssm.step_posterior(h, z, a_prev, embed)

        # Run n_iter MPPI iterations (default 1 for real-time)
        for _ in range(cfg.n_iter):
            self.U = self._mppi_iteration(h_new, z_new)

        # Extract first action
        action = self.U[0].cpu().numpy()

        # ── Warm-start: shift nominal sequence left
        if cfg.warm_start:
            self.U = torch.roll(self.U, -1, dims=0)
            self.U[-1] = 0.0  # zero-pad last step

        info = {
            "best_J":    self.last_best_J,
            "mean_J":    self.last_costs.mean().item() if self.last_costs is not None else None,
            "fallback":  False,
        }

        # ── Fallback check
        if (self.last_costs is not None and
                (self.last_costs < cfg.safety_threshold).sum() == 0):
            action = self._fallback_action(obs)
            info["fallback"] = True
            self.n_fallbacks += 1

        return action, h_new, z_new, info

    def _mppi_iteration(self, h0: torch.Tensor,
                         z0: torch.Tensor) -> torch.Tensor:
        """
        One MPPI iteration: sample K rollouts, compute costs, update U.

        Args:
            h0: (1, deter_dim) — current deterministic state
            z0: (1, stoch_flat) — current stochastic state

        Returns:
            U_new: (H, action_dim) updated nominal action sequence
        """
        cfg = self.cfg
        K, H = cfg.K, cfg.H

        # ── Sample K perturbation sequences: (K, H, action_dim)
        eps = torch.randn(K, H, self.action_dim, device=self.device) * self.sigma

        # Perturbed action sequences (clamp to valid range)
        V = torch.clamp(
            self.U.unsqueeze(0) + eps,   # (K, H, action_dim)
            cfg.action_low, cfg.action_high
        )

        # ── Expand initial latent state to K copies
        h = h0.expand(K, -1).contiguous()   # (K, deter_dim)
        z = z0.expand(K, -1).contiguous()   # (K, stoch_flat)

        # ── Roll out K sequences using PRIOR ONLY (no observations)
        costs = torch.zeros(K, device=self.device)

        for t in range(H):
            # Prior step for all K trajectories in parallel
            h, z, _ = self.agent.rssm.step_prior(h, z, V[:, t, :])

            # Decode latent → predicted obs
            feat         = torch.cat([h, z], -1)           # (K, latent)
            obs_dist     = self.agent.decoder(feat)
            decoded_obs  = symexp(obs_dist.mean)            # (K, obs_dim) — undo symlog

            # Per-step cost
            step_c = self.cost_fn.step_cost(
                decoded_obs, V[:, t, :],
                h=h if cfg.w_conf > 0 else None,
                z=z if cfg.w_conf > 0 else None,
            )
            costs += step_c

        # Terminal cost at end of horizon
        feat        = torch.cat([h, z], -1)
        obs_dist    = self.agent.decoder(feat)
        decoded_obs = symexp(obs_dist.mean)
        costs      += self.cost_fn.terminal_cost(decoded_obs)

        # ── Compute importance weights
        self.last_costs = costs.clone()

        if cfg.normalize_costs:
            # Normalize to [0,1] before applying temperature.
            # This decouples λ from absolute cost scale — critical when
            # adding conformal penalty which may have different magnitude.
            c_min = costs.min(); c_max = costs.max()
            costs_norm = (costs - c_min) / (c_max - c_min + 1e-8)
        else:
            costs_norm = costs

        beta    = costs_norm.min()
        weights = torch.exp(-(costs_norm - beta) / cfg.temperature)
        weights = weights / (weights.sum() + 1e-8)   # (K,)
        self.last_weights = weights.clone()
        self.last_best_J  = costs.min().item()

        # ── Update nominal: U_new_t = U_t + Σ_k w_k · ε_{k,t}
        # weights: (K,)  eps: (K, H, action_dim)
        weighted_eps = (weights.unsqueeze(-1).unsqueeze(-1) * eps).sum(0)  # (H, action_dim)
        U_new = torch.clamp(self.U + weighted_eps, cfg.action_low, cfg.action_high)
        return U_new

    def _fallback_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Emergency brake policy: activate when all K rollouts exceed safety threshold.
        Applies upward thrust to arrest descent, centered laterally.

        This is the fallback required by the project review (Section: Gap 3).
        """
        _, _, z, vx, vy, vz, _ = obs
        # Thrust upward proportionally to descent rate
        Tz = np.clip(-vz / 80.0, 0.5, 1.0)   # strong upward
        Tx = np.clip(-vx / 30.0, -0.3, 0.3)  # dampen lateral
        Ty = np.clip(-vy / 30.0, -0.3, 0.3)
        return np.array([Tx, Ty, Tz], dtype=np.float32)

    def reset(self):
        """Reset nominal action sequence (call at start of each episode)."""
        self._reset_nominal()


# ─────────────────────────────────────────────
# CONFORMAL PREDICTION HOOK (PLACEHOLDER)
# ─────────────────────────────────────────────

class ConformalUncertaintyHook:
    """
    Placeholder for the conformal prediction uncertainty penalty.

    This is the interface that the CP module (Phase 5) will implement.
    It is wired into MPPICostFunction.conf_fn so the planner already
    has the slot — just swap in the real calibrated version later.

    Full CP implementation (Phase 5) will:
      1. Collect calibration episodes (held-out full episodes, not transitions)
      2. Compute nonconformity score: s_t = ||decoder(h,z) - true_obs||_2
      3. Compute quantile q̂ = Quantile({s_t}, (1-α)(1+1/n))
      4. At test time: uncertainty = max(0, s_t_pred - q̂)
      5. This uncertainty is c_conf in the MPPI cost

    Connection to lightning-uq-box:
      Replace _compute_score() with a proper conformal calibrated predictor
      from lightning_uq_box.post_hoc_conformalizers.
    """

    def __init__(self, quantile_threshold: float = 0.5):
        """
        Args:
            quantile_threshold: calibrated nonconformity score threshold.
              Until CP is calibrated, this is a heuristic.
        """
        self.q_hat   = quantile_threshold
        self.enabled = False  # flip to True after CP calibration

    def calibrate(self, agent: DreamerV3, calibration_episodes: list,
                  alpha: float = 0.1, device: str = "cpu"):
        """
        Calibrate the conformal threshold from held-out episodes.

        Args:
            agent:                trained DreamerV3 agent
            calibration_episodes: list of episode dicts (full episodes only)
            alpha:                miscoverage rate (0.1 → 90% coverage)
            device:               torch device

        Sets self.q_hat to the (1-α) quantile of nonconformity scores.
        """
        scores = []
        agent.eval()

        for ep in calibration_episodes:
            obs_seq = torch.FloatTensor(ep["obs"]).to(device)    # (T, obs_dim)
            act_seq = torch.FloatTensor(ep["actions"]).to(device) # (T, action_dim)
            T = obs_seq.shape[0]

            h, z = agent.rssm.initial_state(1, device)
            for t in range(T - 1):
                embed = agent.encoder(obs_seq[t:t+1])
                h, z, _, _ = agent.rssm.step_posterior(
                    h, z, act_seq[t:t+1], embed)

                # Predict next obs from prior step
                h_prior, z_prior, _ = agent.rssm.step_prior(h, z, act_seq[t:t+1])
                feat     = torch.cat([h_prior, z_prior], -1)
                obs_pred = symexp(agent.decoder(feat).mean)

                # Nonconformity score: L2 prediction error (normalized)
                true_next = obs_seq[t+1:t+2]
                score = F.mse_loss(obs_pred, symlog(true_next)).item()
                scores.append(score)

        # Split conformal quantile (corrected for finite calibration set)
        n     = len(scores)
        level = np.ceil((1 - alpha) * (1 + 1/n)) / (1 + 1/n)
        level = np.clip(level, 0, 1)
        self.q_hat   = float(np.quantile(scores, level))
        self.enabled = True
        print(f"  CP calibrated: q̂ = {self.q_hat:.6f}  "
              f"(α={alpha}, n={n} calibration steps)")

    def __call__(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute conformal uncertainty penalty for K latent states.

        Args:
            h: (K, deter_dim)
            z: (K, stoch_flat)

        Returns:
            penalty: (K,) tensor ∈ [0, ∞)
        """
        if not self.enabled:
            return torch.zeros(h.shape[0], device=h.device)
        # Placeholder: use KL-like spread of z as proxy uncertainty.
        # Replace with calibrated conformal score in Phase 5.
        z_reshaped = z.reshape(z.shape[0], -1)
        uncertainty = z_reshaped.var(dim=-1)  # (K,)
        penalty = torch.clamp(uncertainty - self.q_hat, min=0)
        return penalty


# ─────────────────────────────────────────────
# FULL EVALUATION LOOP
# ─────────────────────────────────────────────

def evaluate_mppi(agent: DreamerV3,
                  terrain: TerrainHazardMap,
                  mppi_cfg: MPPIConfig,
                  pdg_cfg: PDGConfig,
                  n_episodes: int = 30,
                  use_conf: bool = False,
                  conf_hook: Optional[ConformalUncertaintyHook] = None,
                  device: str = "cpu") -> dict:
    """
    Evaluate the MPPI planner against the DreamerV3 actor baseline.

    Runs both planners on the same episodes and compares:
      - Success rate
      - Landing position error
      - Landing velocity
      - Terrain hazard at touchdown
      - Fallback activations (MPPI only)

    Args:
        agent:      trained DreamerV3 agent (provides RSSM + encoder + decoder)
        terrain:    TerrainHazardMap for hazard costs
        mppi_cfg:   MPPI configuration
        pdg_cfg:    PDG environment config
        n_episodes: number of evaluation episodes
        use_conf:   whether to enable conformal penalty
        conf_hook:  ConformalUncertaintyHook (must be calibrated if use_conf=True)
        device:     torch device

    Returns:
        dict with "mppi" and "actor" result lists + trajectories
    """
    env = MarsPDGEnv(pdg_cfg)

    # Wire conformal hook into cost function
    conf_fn = conf_hook if (use_conf and conf_hook is not None
                             and conf_hook.enabled) else None
    if use_conf and conf_fn is None:
        print("  ⚠️  Conformal hook not calibrated — running without CP penalty")

    # Build MPPI planner
    cost_fn = MPPICostFunction(mppi_cfg, terrain, pdg_cfg, conf_fn)
    planner = MPPIPlanner(agent, mppi_cfg, cost_fn, device)

    mppi_results, mppi_trajs   = [], []
    actor_results, actor_trajs = [], []

    print(f"\n  Evaluating {n_episodes} episodes each (MPPI vs Actor)...")

    for ep in range(n_episodes):
        seed = 9000 + ep

        # ── MPPI rollout
        planner.reset()
        obs  = env.reset(seed=seed)
        h    = torch.zeros(1, agent.cfg.deter_dim, device=device)
        z    = torch.zeros(1, agent.cfg.stoch_dim * agent.cfg.stoch_classes,
                           device=device)
        done = False; ep_return = 0.0; traj = [obs.copy()]
        n_fb = 0; t0 = time.time()

        while not done:
            action, h, z, info = planner.plan(obs, h, z)
            obs, r, done, ep_info = env.step(action)
            ep_return += r; traj.append(obs.copy())
            if info["fallback"]: n_fb += 1

        plan_time = (time.time() - t0) / len(traj)
        final     = traj[-1]
        hazard_td = terrain.query(final[0], final[1]).item()

        mppi_results.append({
            "success":    ep_info.get("success", False),
            "return":     ep_return,
            "pos_err":    ep_info.get("pos_err", None),
            "vel_err":    ep_info.get("vel_err", None),
            "hazard_td":  hazard_td,
            "n_fallback": n_fb,
            "plan_ms":    plan_time * 1000,
        })
        mppi_trajs.append(np.array(traj))

        # ── DreamerV3 Actor rollout (same seed for fair comparison)
        obs  = env.reset(seed=seed)
        h_a  = torch.zeros(1, agent.cfg.deter_dim, device=device)
        z_a  = torch.zeros(1, agent.cfg.stoch_dim * agent.cfg.stoch_classes,
                           device=device)
        done = False; ep_return_a = 0.0; traj_a = [obs.copy()]

        while not done:
            with torch.no_grad():
                action_a, h_a, z_a = agent.act(obs, h_a, z_a, sample=False)
            obs, r, done, ep_info_a = env.step(action_a)
            ep_return_a += r; traj_a.append(obs.copy())

        final_a   = traj_a[-1]
        hazard_td_a = terrain.query(final_a[0], final_a[1]).item()
        actor_results.append({
            "success":   ep_info_a.get("success", False),
            "return":    ep_return_a,
            "pos_err":   ep_info_a.get("pos_err", None),
            "vel_err":   ep_info_a.get("vel_err", None),
            "hazard_td": hazard_td_a,
        })
        actor_trajs.append(np.array(traj_a))

    # ── Summary
    def _summarize(results, name):
        n_succ    = sum(r["success"] for r in results)
        pos_errs  = [r["pos_err"]  for r in results if r["pos_err"]  is not None]
        vel_errs  = [r["vel_err"]  for r in results if r["vel_err"]  is not None]
        hazards   = [r["hazard_td"] for r in results]
        print(f"\n  [{name}]")
        print(f"    Success:    {n_succ}/{len(results)} "
              f"({100*n_succ/len(results):.0f}%)")
        if pos_errs:
            print(f"    Pos error:  {np.mean(pos_errs):.2f} m  ±{np.std(pos_errs):.2f}")
        if vel_errs:
            print(f"    Vel error:  {np.mean(vel_errs):.2f} m/s")
        print(f"    Hazard @td: {np.mean(hazards):.3f}  (lower=safer)")
        if "n_fallback" in results[0]:
            fb = sum(r["n_fallback"] for r in results)
            print(f"    Fallbacks:  {fb} total")

    _summarize(mppi_results,  "MPPI + Dreamer")
    _summarize(actor_results, "Dreamer Actor")

    return {
        "mppi":        {"results": mppi_results,  "trajectories": mppi_trajs},
        "actor":       {"results": actor_results, "trajectories": actor_trajs},
        "terrain":     terrain,
    }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_comparison(eval_out: dict, pdg_cfg: PDGConfig, save_path: str = None):
    """
    6-panel comparison: terrain hazard map | trajectories | scatter | metrics.
    """
    terrain = eval_out["terrain"]
    mppi_r  = eval_out["mppi"]["results"]
    actor_r = eval_out["actor"]["results"]
    mppi_t  = eval_out["mppi"]["trajectories"]
    actor_t = eval_out["actor"]["trajectories"]

    fig = plt.figure(figsize=(24, 10))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)

    def sax(pos):
        ax = fig.add_subplot(pos, facecolor="#161b22")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        return ax

    ext = terrain.extent

    # ── Terrain hazard map with MPPI trajectories
    ax1 = sax(gs[0, 0:2])
    ax1.imshow(terrain.hazard_map, cmap="RdYlGn_r", vmin=0, vmax=1,
               extent=[-ext, ext, -ext, ext], origin="lower", alpha=0.7)
    for traj, res in zip(mppi_t[:10], mppi_r[:10]):
        ax1.plot(traj[:, 0], traj[:, 1],
                 color="#00ff88" if res["success"] else "#ff6b6b",
                 alpha=0.6, lw=1.2)
    ax1.plot(0, 0, "*", color="#ffff00", markersize=14, label="Target", zorder=5)
    ax1.set_title("MPPI Footprint on Terrain Hazard Map", color="white", fontsize=10)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="white",
               edgecolor="#30363d")

    # ── Descent altitude profiles: MPPI vs Actor
    ax2 = sax(gs[0, 2:4])
    for traj, res in zip(mppi_t[:8], mppi_r[:8]):
        horiz = np.sqrt(traj[:,0]**2 + traj[:,1]**2)
        ax2.plot(horiz, traj[:,2],
                 color="#00d4ff" if res["success"] else "#ff6b6b",
                 lw=1.2, alpha=0.7, label="MPPI" if traj is mppi_t[0] else "")
    for traj, res in zip(actor_t[:8], actor_r[:8]):
        horiz = np.sqrt(traj[:,0]**2 + traj[:,1]**2)
        ax2.plot(horiz, traj[:,2],
                 color="#ffd93d" if res["success"] else "#ff6b6b",
                 lw=1.0, alpha=0.5, ls="--",
                 label="Actor" if traj is actor_t[0] else "")
    ax2.axhline(0, color="white", lw=0.5, ls="--")
    ax2.set_title("Descent Altitude Profiles  (blue=MPPI, yellow=Actor)",
                   color="white", fontsize=10)
    ax2.set_xlabel("Horizontal Distance (m)"); ax2.set_ylabel("Altitude (m)")

    # ── Landing scatter: MPPI
    ax3 = sax(gs[1, 0])
    for r, traj in zip(mppi_r, mppi_t):
        fx, fy = traj[-1, 0], traj[-1, 1]
        ax3.scatter(fx, fy, color="#00ff88" if r["success"] else "#ff6b6b",
                    s=30, alpha=0.8, zorder=3)
    ax3.add_patch(plt.Circle((0,0), pdg_cfg.pos_tol, color="#ffff00",
                               fill=False, lw=1.5, ls="--"))
    ax3.set_aspect("equal")
    ax3.set_title(f"MPPI Landing Scatter", color="white", fontsize=10)
    ax3.set_xlabel("X err (m)"); ax3.set_ylabel("Y err (m)")

    # ── Landing scatter: Actor
    ax4 = sax(gs[1, 1])
    for r, traj in zip(actor_r, actor_t):
        fx, fy = traj[-1, 0], traj[-1, 1]
        ax4.scatter(fx, fy, color="#00ff88" if r["success"] else "#ff6b6b",
                    s=30, alpha=0.8, zorder=3)
    ax4.add_patch(plt.Circle((0,0), pdg_cfg.pos_tol, color="#ffff00",
                               fill=False, lw=1.5, ls="--"))
    ax4.set_aspect("equal")
    ax4.set_title("Actor Landing Scatter", color="white", fontsize=10)
    ax4.set_xlabel("X err (m)"); ax4.set_ylabel("Y err (m)")

    # ── Hazard at touchdown comparison
    ax5 = sax(gs[1, 2])
    ax5.hist([r["hazard_td"] for r in mppi_r], bins=15,
             color="#00d4ff", alpha=0.7, label="MPPI")
    ax5.hist([r["hazard_td"] for r in actor_r], bins=15,
             color="#ffd93d", alpha=0.7, label="Actor")
    ax5.set_title("Terrain Hazard at Touchdown\n(lower=safer)",
                   color="white", fontsize=10)
    ax5.set_xlabel("Hazard Score")
    ax5.legend(fontsize=8, facecolor="#161b22", labelcolor="white",
               edgecolor="#30363d")

    # ── Success / metrics bar chart
    ax6 = sax(gs[1, 3])
    metrics   = ["Success\n(%)", "Avg Hazard\n(×100)", "Fallbacks"]
    mppi_succ = 100 * np.mean([r["success"] for r in mppi_r])
    actr_succ = 100 * np.mean([r["success"] for r in actor_r])
    mppi_haz  = 100 * np.mean([r["hazard_td"] for r in mppi_r])
    actr_haz  = 100 * np.mean([r["hazard_td"] for r in actor_r])
    mppi_fb   = sum(r.get("n_fallback", 0) for r in mppi_r)
    x_ = np.arange(len(metrics))
    ax6.bar(x_ - 0.2, [mppi_succ, mppi_haz, mppi_fb], 0.35,
            color="#00d4ff", alpha=0.8, label="MPPI")
    ax6.bar(x_ + 0.2, [actr_succ, actr_haz, 0],       0.35,
            color="#ffd93d", alpha=0.8, label="Actor")
    ax6.set_xticks(x_); ax6.set_xticklabels(metrics, color="#8b949e", fontsize=8)
    ax6.set_title("Summary Metrics", color="white", fontsize=10)
    ax6.legend(fontsize=8, facecolor="#161b22", labelcolor="white",
               edgecolor="#30363d")

    plt.suptitle(
        "DreamerV3 World Model  |  MPPI Planner vs Actor Baseline  |  Mars PDG",
        color="white", fontsize=12, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run(n_train_steps: int = 50_000,
        n_eval_episodes: int = 30,
        device: str = None):
    """
    Full pipeline:
      1. Train DreamerV3 world model
      2. Build terrain hazard map
      3. Evaluate MPPI vs Actor baseline
      4. (Optional) Calibrate and enable conformal hook
    """
    from dreamerv3_pdg import train_dreamerv3

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pdg_cfg = PDGConfig()

    print("=" * 60)
    print("  Phase 1: Train DreamerV3 world model")
    print("=" * 60)
    agent, _, _ = train_dreamerv3(
        n_env_steps=n_train_steps,
        train_every=5,
        seed_episodes=10,
        device=device,
        plot=False,
    )
    agent.eval()

    print("\n" + "=" * 60)
    print("  Phase 2: Build terrain hazard map")
    print("=" * 60)
    terrain = TerrainHazardMap(size=200, resolution=5.0, seed=42)
    print(f"  Terrain: {terrain.size}×{terrain.size} cells, "
          f"{terrain.resolution}m/cell, "
          f"extent=±{terrain.extent:.0f}m")
    print(f"  Hazard range: [{terrain.hazard_map.min():.3f}, "
          f"{terrain.hazard_map.max():.3f}]")

    print("\n" + "=" * 60)
    print("  Phase 3: MPPI evaluation")
    print("=" * 60)
    mppi_cfg = MPPIConfig(K=512, H=20, temperature=0.05,
                          w_hazard=2.0, w_conf=0.0)

    eval_out = evaluate_mppi(
        agent, terrain, mppi_cfg, pdg_cfg,
        n_episodes=n_eval_episodes,
        use_conf=False,
        device=device,
    )

    plot_comparison(eval_out, pdg_cfg,
                    save_path="mppi_dreamer_output.png")

    # ── Phase 4 hook (fill in when CP is ready)
    print("\n" + "=" * 60)
    print("  Phase 4 (placeholder): Conformal calibration")
    print("=" * 60)
    conf_hook = ConformalUncertaintyHook()
    print("  CP hook created (not yet calibrated).")
    print("  To calibrate: conf_hook.calibrate(agent, cal_episodes, alpha=0.1)")
    print("  Then: run evaluate_mppi(..., use_conf=True, conf_hook=conf_hook)")

    return agent, terrain, eval_out, conf_hook


if __name__ == "__main__":
    """
    Quick start:
        python mppi_dreamer.py

    With conformal (after training):
        agent, terrain, eval_out, conf_hook = run()
        conf_hook.calibrate(agent, cal_episodes)
        eval_out_cp = evaluate_mppi(
            agent, terrain, MPPIConfig(w_conf=1.0), PDGConfig(),
            use_conf=True, conf_hook=conf_hook,
        )

    Tune MPPI:
        MPPIConfig(K=1024, H=25, temperature=0.03)   # more samples, longer horizon
        MPPIConfig(w_hazard=5.0)                      # more hazard-averse
        MPPIConfig(safety_threshold=500.0)            # trigger fallback sooner
    """
    run(n_train_steps=50_000, n_eval_episodes=20)
