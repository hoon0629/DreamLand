"""
DreamerV3 for Mars Powered Descent Guidance (PDG)
===================================================
State-based DreamerV3 — no vision, pure [pos, vel, mass] trajectories
on a MuJoCo-backed 3D lander.

Architecture (faithful to Hafner et al. 2025, Nature):
  ┌─────────────────────────────────────────────────────┐
  │                   WORLD MODEL                       │
  │                                                     │
  │  obs_t ──► Encoder ──► embed_t                      │
  │                           │                         │
  │  h_{t-1}, z_{t-1}, a_{t-1} ──► GRU ──► h_t         │
  │                                          │           │
  │  h_t + embed_t ──► Posterior q(z_t|h_t,x_t)         │
  │  h_t           ──► Prior    p(z_t|h_t)              │
  │                                                     │
  │  [h_t, z_t] ──► Decoder   → obs reconstruction     │
  │  [h_t, z_t] ──► RewardHead → reward prediction     │
  │  [h_t, z_t] ──► ContinueHead → ¬done prediction   │
  └─────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────┐
  │              BEHAVIOUR (Imagination)                │
  │                                                     │
  │  For H steps in latent space (no env interaction):  │
  │  [h_t, z_t] ──► Actor ──► a_t                      │
  │  [h_t, z_t] ──► Critic ──► V_t (two-hot symlog)    │
  │  Update actor to maximize imagined λ-returns        │
  └─────────────────────────────────────────────────────┘

Key DreamerV3 innovations used here:
  - symlog obs normalization (handles large altitude/velocity ranges)
  - KL balancing (α=0.8) + free bits (1 nat)
  - Unimix categoricals (1% uniform)
  - Percentile return normalization (5th/95th)
  - Two-hot symlog value/reward loss
  - RMSNorm + SiLU throughout

Install:
  pip install torch numpy matplotlib scipy gymnasium mujoco

Reference: github.com/danijar/dreamerv3 (JAX original)
           github.com/NM512/dreamerv3-torch (PyTorch port)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import mujoco
except ImportError:
    mujoco = None
from torch.distributions import OneHotCategorical, Normal, Bernoulli
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings
import os
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class DreamerConfig:
    # World model
    obs_dim:       int   = 7       # [x, y, z, vx, vy, vz, mass]
    action_dim:    int   = 3       # [tx_cmd, ty_cmd, throttle_cmd] in [-1, 1]
    embed_dim:     int   = 256     # encoder output dim
    deter_dim:     int   = 512     # GRU hidden (deterministic state h)
    stoch_dim:     int   = 32      # stochastic state z: stoch_dim categories
    stoch_classes: int   = 32      # each category has stoch_classes classes
    hidden_dim:    int   = 512     # MLP hidden size

    # Behaviour
    imag_horizon:  int   = 15      # imagination rollout length
    gamma:         float = 0.997   # discount
    lam:           float = 0.95    # lambda for λ-returns

    # Losses
    kl_balance:    float = 0.8     # weight on prior KL vs posterior KL
    kl_free:       float = 1.0     # free nats (KL clipped below this)
    kl_weight:     float = 1.0
    reward_weight: float = 1.0
    cont_weight:   float = 1.0
    actor_ent:     float = 2e-3    # actor entropy bonus

    # Training
    batch_size:    int   = 16
    seq_len:       int   = 64      # sequence length per batch
    lr_world:      float = 1e-4
    lr_actor:      float = 1e-5
    lr_critic:     float = 1e-4
    grad_clip:     float = 100.0
    buffer_size:   int   = 100_000

    # Unimix
    unimix:        float = 0.01    # 1% uniform mixing for categoricals


@dataclass
class PDGConfig:
    g:         float = 3.72
    mass:      float = 1905.0
    Isp:       float = 225.0
    T_max:     float = 16000.0
    T_min:     float = 0.0
    dt:        float = 0.5
    max_steps: int   = 120
    alt_init:  float = 2000.0
    pos_tol:        float = 50.0   # relaxed for training signal
    pos_tol_strict: float = 5.0    # stricter eval-only landing target
    vel_tol:        float = 2.0


MUJOCO_LANDER_XML = """
<mujoco model="mars_pdg_lander">
  <compiler angle="radian"/>
  <option timestep="0.02" gravity="0 0 -3.72" integrator="RK4"/>
  <size nuserdata="1"/>
  <visual>
    <map znear="0.01"/>
  </visual>
  <worldbody>
    <light pos="0 0 10" dir="0 0 -1"/>
    <geom name="ground" type="plane" pos="0 0 0" size="5000 5000 0.1"
          rgba="0.3 0.25 0.2 1" friction="1.0 0.1 0.1"/>
    <body name="lander" pos="0 0 0">
      <joint name="x" type="slide" axis="1 0 0" damping="0.2"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0.2"/>
      <joint name="z" type="slide" axis="0 0 1" damping="0.2"/>
      <geom name="lander_geom" type="sphere" size="0.5" mass="1905"
            rgba="0.8 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


# ─────────────────────────────────────────────
# PDG ENVIRONMENT
# ─────────────────────────────────────────────

class MarsPDGEnv:
    """3D MuJoCo Mars powered descent. Action = [tx_cmd, ty_cmd, throttle_cmd]."""

    def __init__(self, cfg: PDGConfig = None):
        if mujoco is None:
            raise ImportError(
                "MuJoCo is required for MarsPDGEnv. Install with `pip install mujoco`."
            )
        self.cfg = cfg or PDGConfig()
        self.model = mujoco.MjModel.from_xml_string(MUJOCO_LANDER_XML)
        self.data = mujoco.MjData(self.model)
        self.lander_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "lander"
        )
        self.lander_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "lander_geom"
        )
        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )
        self.lander_radius = float(self.model.geom_size[self.lander_geom_id][0])
        self.frame_skip = max(1, int(round(self.cfg.dt / self.model.opt.timestep)))
        self.g0 = 9.81
        self.mass = self.cfg.mass
        self.step_count = 0
        self.training_steps = 0   # tracked by training loop for curriculum/debugging

    def reset(self, seed=None, difficulty=0.0):
        if seed is not None: np.random.seed(seed)
        c = self.cfg
        d = float(np.clip(difficulty, 0.0, 1.0))
        x_range = 50.0 + 150.0 * d
        y_range = 50.0 + 150.0 * d
        z_range = 30.0 + 70.0 * d
        vx_range = 5.0 + 25.0 * d
        vy_range = 5.0 + 25.0 * d
        vz_low = -10.0 - 70.0 * d
        vz_high = -3.0 - 17.0 * d
        x = np.random.uniform(-x_range, x_range)
        y = np.random.uniform(-y_range, y_range)
        z = c.alt_init + np.random.uniform(-z_range, z_range)
        vx = np.random.uniform(-vx_range, vx_range)
        vy = np.random.uniform(-vy_range, vy_range)
        vz = np.random.uniform(vz_low, vz_high)
        self.mass = c.mass * np.random.uniform(0.5, 0.8)
        self.data.qpos[:] = np.array([x, y, z + self.lander_radius], dtype=np.float64)
        self.data.qvel[:] = np.array([vx, vy, vz], dtype=np.float64)
        self._set_mass(self.mass)
        self.data.xfrc_applied.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        c = self.cfg
        x, y, z, vx, vy, vz, mass = self._get_state()
        horiz_pos = np.hypot(x, y)
        horiz_vel = np.hypot(vx, vy)
        prev_goal_cost = (
            0.02 * horiz_pos +
            0.005 * max(z, 0.0) +
            0.02 * horiz_vel +
            0.04 * abs(vz)
        )
        action = np.clip(action, -1, 1)
        tx_cmd = action[0]
        ty_cmd = action[1]
        throttle_cmd = 0.5 * (action[2] + 1.0)  # [-1, 1] -> [0, 1]
        Tx = tx_cmd * 0.3 * c.T_max
        Ty = ty_cmd * 0.3 * c.T_max
        Tz = c.T_max * (c.T_min + throttle_cmd * (1.0 - c.T_min))

        T_norm = np.linalg.norm([Tx, Ty, Tz])
        dm = T_norm / (c.Isp * self.g0) * c.dt
        self.data.xfrc_applied[self.lander_body_id, :3] = np.array([Tx, Ty, Tz], dtype=np.float64)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[self.lander_body_id, :3] = 0.0
        self.mass = max(mass - dm, c.mass * 0.1)
        self._set_mass(self.mass)
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1

        x, y, z, vx, vy, vz, mass = self._get_state()
        pos_err = np.hypot(x, y)
        horiz_vel = np.hypot(vx, vy)
        vel_err = np.sqrt(vx**2 + vy**2 + vz**2)
        goal_cost = (
            0.02 * pos_err +
            0.005 * max(z, 0.0) +
            0.02 * horiz_vel +
            0.04 * abs(vz)
        )
        fuel_pen = T_norm / (c.T_max * 200.0)
        lat_thrust_pen = np.sqrt(Tx**2 + Ty**2) / (c.T_max * 300.0)
        reward = prev_goal_cost - goal_cost - fuel_pen - lat_thrust_pen

        done = False; info = {}
        # Termination 1: touched the ground
        if self._has_ground_contact() or z <= 0:
            done = True
            success_relaxed = pos_err < c.pos_tol and vel_err < c.vel_tol
            success_strict = pos_err < c.pos_tol_strict and vel_err < c.vel_tol
            touchdown_score = 180.0 - 0.05 * pos_err - 3.0 * horiz_vel - 1.5 * abs(vz)
            reward += touchdown_score
            if success_relaxed:
                reward += 150.0
            info = {"success": success_relaxed,
                    "success_relaxed": success_relaxed,
                    "success_strict": success_strict,
                    "pos_err": pos_err, "vel_err": vel_err,
                    "x_err": x, "y_err": y}

        # Termination 2: flew too high
        elif z > c.alt_init * 1.5:
            done = True
            reward -= 300.0 + 0.05 * pos_err + 0.05 * max(z, 0.0)
            info = {"success": False, "flew_away": True}

        # Termination 3: drifted too far horizontally
        elif pos_err > 2000.0:
            done = True
            reward -= 250.0 + 0.05 * pos_err
            info = {"success": False, "drifted": True}

        elif self.step_count >= c.max_steps:
            done = True
            reward -= 300.0 + 0.05 * pos_err + 0.05 * max(z, 0.0)
            info = {"timeout": True, "success": False}

        return self._get_state(), float(reward), done, info

    def _set_mass(self, mass: float):
        self.model.body_mass[self.lander_body_id] = mass

    def _get_state(self):
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        z = max(0.0, float(self.data.qpos[2]) - self.lander_radius)
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        vz = float(self.data.qvel[2])
        return np.array([x, y, z, vx, vy, vz, self.mass], dtype=np.float32)

    def _has_ground_contact(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = {int(contact.geom1), int(contact.geom2)}
            if self.lander_geom_id in geoms and self.ground_geom_id in geoms:
                return True
        return False


# ─────────────────────────────────────────────
# SYMLOG TRANSFORMS (DreamerV3 key innovation)
# ─────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform — handles large-magnitude states like altitude."""
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Two-hot encoding for symlog value targets.
    Distributes value across two adjacent bins proportionally.
    """
    x = x.unsqueeze(-1)
    below = (bins <= x).sum(-1) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1
    frac = (x.squeeze(-1) - bins[below]) / (bins[above] - bins[below] + 1e-8)
    target = torch.zeros(*x.shape[:-1], len(bins), device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), (1 - frac).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), frac.unsqueeze(-1))
    return target


# ─────────────────────────────────────────────
# NETWORK BUILDING BLOCKS
# ─────────────────────────────────────────────

def mlp(in_dim: int, out_dim: int, hidden: int, layers: int = 2,
        norm: bool = True, act=nn.SiLU) -> nn.Sequential:
    """MLP with RMSNorm + SiLU (DreamerV3 standard)."""
    dims = [in_dim] + [hidden] * layers + [out_dim]
    modules = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        modules.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 2:
            if norm: modules.append(nn.RMSNorm(d_out))
            modules.append(act())
    return nn.Sequential(*modules)


class CategoricalStraightThrough(nn.Module):
    """
    Straight-through categorical for stochastic state z.
    Outputs one-hot samples with gradients passed through softmax.
    Includes unimix (1% uniform) for exploration stability.
    """
    def __init__(self, in_dim: int, stoch_dim: int, stoch_classes: int,
                 unimix: float = 0.01):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.unimix = unimix
        self.fc = nn.Linear(in_dim, stoch_dim * stoch_classes)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(x).reshape(*x.shape[:-1], self.stoch_dim, self.stoch_classes)
        # Unimix: mix logits with uniform (1%)
        probs = torch.softmax(logits, -1)
        probs = (1 - self.unimix) * probs + self.unimix / self.stoch_classes
        # Straight-through: sample in forward, use softmax for gradients
        dist = OneHotCategorical(probs=probs)
        sample = dist.sample()
        sample_sg = sample + probs - probs.detach()  # straight-through
        flat = sample_sg.reshape(*x.shape[:-1], self.stoch_dim * self.stoch_classes)
        return flat, logits


# ─────────────────────────────────────────────
# RSSM — Recurrent State Space Model
# ─────────────────────────────────────────────

class RSSM(nn.Module):
    """
    Core of DreamerV3 world model.

    Latent state = (h_t, z_t)
      h_t: deterministic GRU hidden state (memory / running context)
      z_t: stochastic categorical state (uncertainty about current state)

    Prior:     p(z_t | h_t)              — used during imagination
    Posterior: q(z_t | h_t, embed_t)    — used during world model training
    """

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        c = cfg
        self.deter_dim    = c.deter_dim
        self.stoch_dim    = c.stoch_dim
        self.stoch_classes = c.stoch_classes
        self.latent_dim   = c.deter_dim + c.stoch_dim * c.stoch_classes

        # GRU input: [z_{t-1} flat, action_t]
        gru_in = c.stoch_dim * c.stoch_classes + c.action_dim
        self.gru_norm = nn.RMSNorm(c.deter_dim)
        self.gru = nn.GRUCell(gru_in, c.deter_dim)

        # Prior: h_t → z_t
        self.prior_net = nn.Sequential(
            nn.Linear(c.deter_dim, c.hidden_dim), nn.RMSNorm(c.hidden_dim), nn.SiLU(),
        )
        self.prior_head = CategoricalStraightThrough(
            c.hidden_dim, c.stoch_dim, c.stoch_classes, c.unimix)

        # Posterior: [h_t, embed_t] → z_t
        self.post_net = nn.Sequential(
            nn.Linear(c.deter_dim + c.embed_dim, c.hidden_dim),
            nn.RMSNorm(c.hidden_dim), nn.SiLU(),
        )
        self.post_head = CategoricalStraightThrough(
            c.hidden_dim, c.stoch_dim, c.stoch_classes, c.unimix)

    def initial_state(self, batch: int, device):
        h = torch.zeros(batch, self.deter_dim, device=device)
        z = torch.zeros(batch, self.stoch_dim * self.stoch_classes, device=device)
        return h, z

    def step_prior(self, h: torch.Tensor, z: torch.Tensor,
                   action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step with prior (imagination — no observation)."""
        x = torch.cat([z, action], -1)
        h = self.gru(x, h)
        h = self.gru_norm(h)
        feat = self.prior_net(h)
        z_flat, logits = self.prior_head(feat)
        return h, z_flat, logits

    def step_posterior(self, h: torch.Tensor, z: torch.Tensor,
                       action: torch.Tensor,
                       embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor, torch.Tensor]:
        """One step with posterior (using actual observation embed)."""
        x = torch.cat([z, action], -1)
        h = self.gru(x, h)
        h = self.gru_norm(h)
        # Prior logits
        prior_feat = self.prior_net(h)
        _, prior_logits = self.prior_head(prior_feat)
        # Posterior logits (uses embed)
        post_feat = self.post_net(torch.cat([h, embed], -1))
        z_flat, post_logits = self.post_head(post_feat)
        return h, z_flat, prior_logits, post_logits

    def observe_sequence(self, embeds: torch.Tensor,
                         actions: torch.Tensor) -> dict:
        """
        Process a full sequence to get latent states.
        embeds:  (T, B, embed_dim)
        actions: (T, B, action_dim)  — a_{t-1} leading to obs_t
        """
        T, B = embeds.shape[:2]
        device = embeds.device
        h, z = self.initial_state(B, device)

        hs, zs, prior_logits_list, post_logits_list = [], [], [], []

        for t in range(T):
            h, z, prior_logits, post_logits = self.step_posterior(
                h, z, actions[t], embeds[t])
            hs.append(h); zs.append(z)
            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)

        return {
            "h":            torch.stack(hs),            # (T, B, deter)
            "z":            torch.stack(zs),            # (T, B, stoch_flat)
            "prior_logits": torch.stack(prior_logits_list),  # (T, B, stoch, classes)
            "post_logits":  torch.stack(post_logits_list),
        }

    def imagine_sequence(self, h0: torch.Tensor, z0: torch.Tensor,
                         actor, horizon: int) -> dict:
        """
        Imagination rollout using prior only (no observations).
        Actor samples actions from current latent state.
        """
        h, z = h0, z0
        hs, zs, actions, log_probs = [], [], [], []

        for _ in range(horizon):
            feat = torch.cat([h, z], -1)  # latent feature
            action, lp = actor(feat)
            h, z, _ = self.step_prior(h, z, action)
            hs.append(h); zs.append(z)
            actions.append(action); log_probs.append(lp)

        return {
            "h":         torch.stack(hs),
            "z":         torch.stack(zs),
            "actions":   torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "feats":     torch.stack([torch.cat([h, z], -1)
                                      for h, z in zip(hs, zs)]),
        }


# ─────────────────────────────────────────────
# WORLD MODEL COMPONENTS
# ─────────────────────────────────────────────

class StateEncoder(nn.Module):
    """MLP encoder: symlog(obs) → embed. Replaces CNN for state-based tasks."""
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.net = mlp(cfg.obs_dim, cfg.embed_dim,
                       cfg.hidden_dim, layers=2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(symlog(obs))


class StateDecoder(nn.Module):
    """MLP decoder: latent → symlog(obs_pred)."""
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.net = mlp(cfg.deter_dim + cfg.stoch_dim * cfg.stoch_classes,
                       cfg.obs_dim, cfg.hidden_dim, layers=2)

    def forward(self, feat: torch.Tensor) -> Normal:
        mean = self.net(feat)
        return Normal(mean, torch.ones_like(mean))


class RewardHead(nn.Module):
    """Predicts symlog-transformed reward with two-hot categorical output."""
    def __init__(self, cfg: DreamerConfig, n_bins: int = 255):
        super().__init__()
        self.n_bins = n_bins
        self.net = mlp(cfg.deter_dim + cfg.stoch_dim * cfg.stoch_classes,
                       n_bins, cfg.hidden_dim, layers=2)
        # Bins live in symlog space; predictions are mapped back with symexp.
        bins = torch.linspace(-10, 10, n_bins)
        self.register_buffer("bins", bins)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns predicted reward (scalar)."""
        logits = self.net(feat)
        probs = torch.softmax(logits, -1)
        pred_symlog = (probs * self.bins).sum(-1)
        return symexp(pred_symlog)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Two-hot symlog loss."""
        logits = self.net(feat)
        target_symlog = symlog(target)
        target_twohot = twohot_encode(target_symlog, self.bins)
        return -(target_twohot * F.log_softmax(logits, -1)).sum(-1).mean()


class ContinueHead(nn.Module):
    """Predicts P(episode continues) — Bernoulli."""
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.net = mlp(cfg.deter_dim + cfg.stoch_dim * cfg.stoch_classes,
                       1, cfg.hidden_dim, layers=1)

    def forward(self, feat: torch.Tensor) -> Bernoulli:
        logit = self.net(feat).squeeze(-1)
        return Bernoulli(logits=logit)


# ─────────────────────────────────────────────
# ACTOR & CRITIC
# ─────────────────────────────────────────────

class Actor(nn.Module):
    """
    Continuous action actor.
    Outputs tanh-squashed Normal distribution over [tx_cmd, ty_cmd, throttle_cmd].
    """
    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        latent = cfg.deter_dim + cfg.stoch_dim * cfg.stoch_classes
        self.net = mlp(latent, cfg.hidden_dim, cfg.hidden_dim, layers=3)
        self.mean_head = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.std_head  = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.std_head.weight)
        nn.init.zeros_(self.std_head.bias)
        self.mean_head.bias.data[2] = -2.0  # low initial throttle bias

    def forward(self, feat: torch.Tensor,
                sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(feat)
        mean = self.mean_head(h)
        std  = F.softplus(self.std_head(h)) + 0.1  # minimum std
        dist = Normal(mean, std)
        action_raw = dist.rsample() if sample else mean
        action = torch.tanh(action_raw)  # squash to [-1, 1]
        # Log prob with tanh squashing correction
        log_prob = dist.log_prob(action_raw).sum(-1) - \
                   torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob


class Critic(nn.Module):
    """
    Value critic with two-hot symlog output (DreamerV3 style).
    Estimates discounted λ-return from latent state.
    """
    def __init__(self, cfg: DreamerConfig, n_bins: int = 255):
        super().__init__()
        latent = cfg.deter_dim + cfg.stoch_dim * cfg.stoch_classes
        self.n_bins = n_bins
        self.net = mlp(latent, n_bins, cfg.hidden_dim, layers=3)
        bins = torch.linspace(-10, 10, n_bins)
        self.register_buffer("bins", bins)
        # Zero-initialize last layer (helps early training)
        nn.init.zeros_(list(self.net.children())[-1].weight)
        nn.init.zeros_(list(self.net.children())[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.net(feat)
        probs = torch.softmax(logits, -1)
        pred_symlog = (probs * self.bins).sum(-1)
        return symexp(pred_symlog)  # expected value in original space

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.net(feat)
        target_symlog = symlog(target)
        target_twohot = twohot_encode(target_symlog, self.bins)
        return -(target_twohot * F.log_softmax(logits, -1)).sum(-1).mean()


# ─────────────────────────────────────────────
# DREAMERV3 AGENT
# ─────────────────────────────────────────────

class DreamerV3(nn.Module):
    """
    DreamerV3 agent for state-based PDG.
    Integrates world model + actor + critic with all DreamerV3 training details.
    """

    def __init__(self, cfg: DreamerConfig = None):
        super().__init__()
        self.cfg = cfg or DreamerConfig()
        c = self.cfg

        # World model
        self.encoder  = StateEncoder(c)
        self.rssm     = RSSM(c)
        self.decoder  = StateDecoder(c)
        self.reward_h = RewardHead(c)
        self.cont_h   = ContinueHead(c)

        # Behaviour
        self.actor    = Actor(c)
        self.critic   = Critic(c)

        # Optimizers (separate lr per component as in paper)
        wm_params = (list(self.encoder.parameters()) +
                     list(self.rssm.parameters()) +
                     list(self.decoder.parameters()) +
                     list(self.reward_h.parameters()) +
                     list(self.cont_h.parameters()))
        self.opt_wm     = torch.optim.Adam(wm_params, lr=c.lr_world, eps=1e-8)
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=c.lr_actor, eps=1e-8)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=c.lr_critic, eps=1e-8)

        # Percentile normalization buffers (return normalization)
        self.return_ema_low  = None
        self.return_ema_high = None

    @property
    def device(self):
        return next(self.parameters()).device

    def latent_feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], -1)

    # ── World Model Loss ──────────────────────────────────────

    def world_model_loss(self, obs: torch.Tensor, actions: torch.Tensor,
                         rewards: torch.Tensor, dones: torch.Tensor) -> dict:
        """
        obs:     (T, B, obs_dim)
        actions: (T, B, action_dim)
        rewards: (T, B)
        dones:   (T, B)
        """
        T, B = obs.shape[:2]

        # Encode all observations
        embeds = self.encoder(obs.reshape(T*B, -1)).reshape(T, B, -1)

        # RSSM expects a_{t-1} for obs_t. Replay stores a_t alongside obs_t,
        # so shift actions by one and prepend zeros at sequence start.
        prev_actions = torch.cat([
            torch.zeros(1, B, self.cfg.action_dim, device=actions.device, dtype=actions.dtype),
            actions[:-1],
        ], dim=0)

        # RSSM forward pass
        rssm_out = self.rssm.observe_sequence(embeds, prev_actions)
        h = rssm_out["h"]; z = rssm_out["z"]
        feat = self.latent_feat(h, z)

        # ── Reconstruction loss (symlog MSE)
        obs_dist = self.decoder(feat)
        recon_loss = -obs_dist.log_prob(symlog(obs)).mean()

        # ── Reward prediction loss (two-hot symlog)
        reward_loss = self.reward_h.loss(feat, rewards)

        # ── Continue prediction loss (Bernoulli)
        cont_dist = self.cont_h(feat)
        cont_loss = -cont_dist.log_prob(1.0 - dones).mean()

        # ── KL loss (prior vs posterior) with balancing
        prior_logits = rssm_out["prior_logits"]   # (T, B, stoch, classes)
        post_logits  = rssm_out["post_logits"]

        # KL balancing with stop-gradients:
        # α * KL(post_sg || prior) + (1-α) * KL(post || prior_sg)
        post_probs  = torch.softmax(post_logits, -1)
        prior_probs = torch.softmax(prior_logits, -1)
        post_probs_sg = post_probs.detach()
        prior_probs_sg = prior_probs.detach()
        kl_lhs = (post_probs_sg * (torch.log(post_probs_sg + 1e-8) -
                                   torch.log(prior_probs + 1e-8))).sum(-1).sum(-1)
        kl_rhs = (post_probs * (torch.log(post_probs + 1e-8) -
                                torch.log(prior_probs_sg + 1e-8))).sum(-1).sum(-1)

        alpha = self.cfg.kl_balance
        kl_loss = (alpha * kl_lhs.mean() +
                   (1 - alpha) * kl_rhs.mean())
        # Free bits
        kl_loss = torch.clamp(kl_loss, min=self.cfg.kl_free)

        total = (recon_loss +
                 self.cfg.kl_weight * kl_loss +
                 self.cfg.reward_weight * reward_loss +
                 self.cfg.cont_weight * cont_loss)

        return {
            "total": total, "recon": recon_loss, "kl": kl_loss,
            "reward": reward_loss, "cont": cont_loss,
            "h": h.detach(), "z": z.detach(),
        }

    # ── Behaviour Loss (Imagination) ─────────────────────────

    def behaviour_loss(self, h0: torch.Tensor, z0: torch.Tensor) -> dict:
        """
        Train actor-critic purely from imagined trajectories.
        No environment interaction during this step.
        """
        cfg = self.cfg
        H = cfg.imag_horizon

        # Flatten batch dims for imagination start
        B = h0.shape[0] * h0.shape[1]
        h0 = h0.reshape(B, -1)
        z0 = z0.reshape(B, -1)

        # Imagination rollout
        imag = self.rssm.imagine_sequence(h0, z0, self.actor, H)
        feats   = imag["feats"]    # (H, B, latent)
        actions = imag["actions"]  # (H, B, action)

        # Predict rewards and continues along imagined trajectory
        rewards = self.reward_h(feats)         # (H, B)
        conts   = self.cont_h(feats).probs     # (H, B)

        # ── λ-returns (DreamerV3 style)
        values = self.critic(feats).detach()   # (H, B)
        returns = self._lambda_returns(rewards, conts, values, cfg.gamma, cfg.lam)

        # ── Percentile return normalization (5th / 95th percentile)
        ret_flat = returns.detach().reshape(-1)
        low  = torch.quantile(ret_flat, 0.05)
        high = torch.quantile(ret_flat, 0.95)
        if self.return_ema_low is None:
            self.return_ema_low  = low
            self.return_ema_high = high
        else:
            self.return_ema_low  = 0.99 * self.return_ema_low  + 0.01 * low
            self.return_ema_high = 0.99 * self.return_ema_high + 0.01 * high
        scale = torch.clamp(self.return_ema_high - self.return_ema_low, min=1.0)
        norm_returns = (returns - self.return_ema_low) / scale

        # ── Actor loss: maximize normalized returns + entropy
        log_probs = imag["log_probs"]  # (H, B)
        actor_loss = -(norm_returns[:-1].detach() * log_probs[:-1]).mean()
        ent_loss   = -log_probs[:-1].mean() * cfg.actor_ent
        actor_total = actor_loss + ent_loss

        # ── Critic loss: two-hot symlog
        critic_loss = self.critic.loss(feats[:-1].detach(), returns[:-1].detach())

        return {
            "actor": actor_total,
            "critic": critic_loss,
            "mean_return": returns.mean().item(),
            "mean_reward": rewards.mean().item(),
        }

    @staticmethod
    def _lambda_returns(rewards, conts, values, gamma, lam):
        """Compute λ-returns for imagined trajectory."""
        H = rewards.shape[0]
        last_val = values[-1]
        returns = torch.zeros_like(rewards)
        for t in reversed(range(H - 1)):
            bootstrap = (1 - lam) * values[t] + lam * last_val
            returns[t] = rewards[t] + gamma * conts[t] * bootstrap
            last_val = returns[t]
        returns[-1] = values[-1]
        return returns

    # ── Training Step ─────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        """
        Full DreamerV3 training step on one batch.
        batch: {obs, actions, rewards, dones} each (T, B, *)
        """
        obs     = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones   = batch["dones"].to(self.device)

        # 1. World model
        self.opt_wm.zero_grad()
        wm_loss = self.world_model_loss(obs, actions, rewards, dones)
        wm_loss["total"].backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.parameters() if p.grad is not None],
            self.cfg.grad_clip)
        self.opt_wm.step()

        # 2. Behaviour (from imagined trajectories seeded by RSSM states)
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        beh_loss = self.behaviour_loss(wm_loss["h"], wm_loss["z"])
        beh_loss["actor"].backward()
        beh_loss["critic"].backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),  self.cfg.grad_clip)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.opt_actor.step()
        self.opt_critic.step()

        return {**{k: v.item() for k, v in wm_loss.items()
                   if isinstance(v, torch.Tensor) and v.numel() == 1},
                **{k: v for k, v in beh_loss.items()
                   if not isinstance(v, torch.Tensor)},
                "actor_loss":  beh_loss["actor"].item(),
                "critic_loss": beh_loss["critic"].item()}

    # ── Action Selection ──────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray, h: torch.Tensor,
            z: torch.Tensor, prev_action: Optional[np.ndarray] = None,
            sample: bool = True):
        """
        Select action given current observation and RSSM state.
        Returns (action, new_h, new_z) — h, z must be tracked across steps.
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if prev_action is None:
            prev_action_t = torch.zeros(1, self.cfg.action_dim, device=self.device)
        else:
            prev_action_t = torch.as_tensor(prev_action, dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
        embed = self.encoder(obs_t)
        h, z, _, _ = self.rssm.step_posterior(h, z,
                                               prev_action_t,
                                               embed)
        feat = self.latent_feat(h, z)
        action, _ = self.actor(feat, sample=sample)
        return action.squeeze(0).cpu().numpy(), h, z


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────

class EpisodeBuffer:
    """
    Stores complete episodes. Samples random subsequences for training.
    DreamerV3 trains on sequences (not single transitions) to learn temporal structure.
    """

    def __init__(self, capacity: int = 100_000, seq_len: int = 64):
        self.capacity = capacity
        self.seq_len  = seq_len
        self.episodes = deque()
        self.total_steps = 0

    def add_episode(self, episode: dict):
        """episode: {obs, actions, rewards, dones} each (T, *)"""
        T = len(episode["obs"])
        if T < self.seq_len:
            return  # skip too-short episodes
        self.episodes.append(episode)
        self.total_steps += T
        while self.total_steps > self.capacity and self.episodes:
            removed = self.episodes.popleft()
            self.total_steps -= len(removed["obs"])

    def sample(self, batch_size: int, seq_len: int) -> dict:
        """Sample a batch of (seq_len, batch_size, *) tensors."""
        obs_l, act_l, rew_l, don_l = [], [], [], []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            T = len(ep["obs"])
            start = random.randint(0, T - seq_len)
            obs_l.append(ep["obs"]   [start:start+seq_len])
            act_l.append(ep["actions"][start:start+seq_len])
            rew_l.append(ep["rewards"][start:start+seq_len])
            don_l.append(ep["dones"]  [start:start+seq_len])

        return {
            "obs":     torch.FloatTensor(np.stack(obs_l,  1)),  # (T, B, obs)
            "actions": torch.FloatTensor(np.stack(act_l,  1)),  # (T, B, act)
            "rewards": torch.FloatTensor(np.stack(rew_l,  1)),  # (T, B)
            "dones":   torch.FloatTensor(np.stack(don_l,  1)),  # (T, B)
        }

    def __len__(self):
        return self.total_steps


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def curriculum_difficulty(total_steps: int) -> float:
    """Stage difficulty so the agent learns centering on easy starts first."""
    if total_steps < 50_000:
        return 0.0
    if total_steps < 100_000:
        return 0.2
    if total_steps < 150_000:
        return 0.4
    return min(1.0, 0.4 + (total_steps - 150_000) / 150_000 * 0.6)


def pd_guidance_action(obs: np.ndarray, cfg: PDGConfig) -> np.ndarray:
    """Simple classical PD baseline in the same action space as the actor."""
    x, y, z, vx, vy, vz, mass = obs
    horiz_kp = 0.015
    horiz_kd = 0.12
    vz_target = -20.0 if z > 800.0 else (-8.0 if z > 100.0 else -2.0)
    vert_kp = 0.0
    vert_kd = 0.35

    ax_des = -horiz_kp * x - horiz_kd * vx
    ay_des = -horiz_kp * y - horiz_kd * vy
    az_des = -vert_kp * max(z, 0.0) + vert_kd * (vz_target - vz)

    max_lat_acc = 0.3 * cfg.T_max / max(mass, cfg.mass * 0.1)
    tx_cmd = np.clip(ax_des / max(max_lat_acc, 1e-6), -1.0, 1.0)
    ty_cmd = np.clip(ay_des / max(max_lat_acc, 1e-6), -1.0, 1.0)

    thrust_z = mass * (cfg.g + az_des)
    throttle = np.clip((thrust_z / cfg.T_max - cfg.T_min) / max(1.0 - cfg.T_min, 1e-6), 0.0, 1.0)
    throttle_cmd = 2.0 * throttle - 1.0
    return np.array([tx_cmd, ty_cmd, throttle_cmd], dtype=np.float32)

def train_dreamerv3(n_env_steps: int = 100_000,
                    train_every: int = 5,
                    seed_episodes: int = 10,
                    device: str = None,
                    plot: bool = True,
                    resume_from: str = None):
    """
    Main DreamerV3 training loop for Mars PDG.

    Steps per iteration:
      1. Collect experience from env using current actor
      2. Every `train_every` env steps: sample batch → train_step
      3. Track metrics

    Args:
        n_env_steps:   total environment steps
        train_every:   train world model every N env steps
        seed_episodes: random episodes before learning starts
        device:        "cuda" / "cpu"
        plot:          show training curves + eval trajectories
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 DreamerV3 — Mars PDG Training")
    print(f"   device={device}  steps={n_env_steps:,}  train_every={train_every}")
    print("=" * 55)

    pdg_cfg    = PDGConfig()
    dream_cfg  = DreamerConfig()
    env        = MarsPDGEnv(pdg_cfg)
    agent      = DreamerV3(dream_cfg).to(device)
    buffer     = EpisodeBuffer(dream_cfg.buffer_size, dream_cfg.seq_len)

    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        print(f"  Resumed from {resume_from}")
        print(f"  Previous run: ep={ckpt['ep_num']}, steps={ckpt['total_steps']}")

    metrics = {"wm_loss": [], "actor_loss": [], "critic_loss": [],
               "reward": [], "ep_return": [], "success_rate": [],
               "env_steps": []}
    recent_returns = deque(maxlen=20)
    recent_success = deque(maxlen=20)
    total_steps = 0
    n_updates = 0

    # ── Seed episodes (random actions)
    print(f"\n  Collecting {seed_episodes} seed episodes...")
    for i in range(seed_episodes):
        obs = env.reset(seed=i, difficulty=0.0)
        ep = {"obs": [], "actions": [], "rewards": [], "dones": []}
        done = False
        while not done:
            action = np.random.uniform(-1, 1, dream_cfg.action_dim)
            next_obs, r, done, _ = env.step(action)
            ep["obs"].append(obs); ep["actions"].append(action)
            ep["rewards"].append(r); ep["dones"].append(float(done))
            obs = next_obs
            total_steps += 1
        for k in ep: ep[k] = np.array(ep[k])
        buffer.add_episode(ep)

    print(f"  Buffer: {len(buffer):,} transitions from {seed_episodes} episodes")

    # ── Main loop
    print(f"\n  Starting main training loop...")
    ep_num = 0
    h = z = None  # RSSM state across episode

    env.training_steps = 0
    obs = env.reset(seed=1000, difficulty=0.0)
    h = torch.zeros(1, dream_cfg.deter_dim, device=device)
    z = torch.zeros(1, dream_cfg.stoch_dim * dream_cfg.stoch_classes, device=device)
    prev_action = np.zeros(dream_cfg.action_dim, dtype=np.float32)
    ep = {"obs": [], "actions": [], "rewards": [], "dones": []}
    ep_return = 0.0

    while total_steps < n_env_steps:
        # Collect one step
        action, h, z = agent.act(obs, h, z, prev_action=prev_action, sample=True)
        next_obs, r, done, info = env.step(action)
        ep["obs"].append(obs); ep["actions"].append(action)
        ep["rewards"].append(r); ep["dones"].append(float(done))
        obs = next_obs
        prev_action = action
        ep_return += r
        total_steps += 1

        if done:
            for k in ep: ep[k] = np.array(ep[k])
            buffer.add_episode(ep)
            recent_returns.append(ep_return)
            recent_success.append(float(info.get("success", False)))
            ep_num += 1

            if ep_num % 20 == 0:
                print(f"  ep={ep_num:4d}  steps={total_steps:6d}  "
                      f"return={np.mean(recent_returns):7.1f}  "
                      f"success={np.mean(recent_success)*100:4.0f}%  "
                      f"updates={n_updates}")
                
            if ep_num % 100 == 0 and ep_num > 0:
                torch.save({
                    "agent":       agent.state_dict(),
                    "ep_num":      ep_num,
                    "total_steps": total_steps,
                    "metrics":     metrics,
                }, f"dreamerv3_ckpt_ep{ep_num}.pth")
                print(f"  💾 Checkpoint → dreamerv3_ckpt_ep{ep_num}.pth")

            difficulty = curriculum_difficulty(total_steps)
            env.training_steps = total_steps
            obs = env.reset(difficulty=difficulty)
            h = torch.zeros(1, dream_cfg.deter_dim, device=device)
            z = torch.zeros(1, dream_cfg.stoch_dim * dream_cfg.stoch_classes,
                            device=device)
            prev_action = np.zeros(dream_cfg.action_dim, dtype=np.float32)
            ep = {"obs": [], "actions": [], "rewards": [], "dones": []}
            ep_return = 0.0

        # Train
        if total_steps % train_every == 0 and len(buffer) >= dream_cfg.batch_size * dream_cfg.seq_len:
            batch = buffer.sample(dream_cfg.batch_size, dream_cfg.seq_len)
            loss_dict = agent.train_step(batch)
            n_updates += 1

            if n_updates % 100 == 0:
                metrics["wm_loss"].append(loss_dict.get("total", 0))
                metrics["actor_loss"].append(loss_dict.get("actor_loss", 0))
                metrics["critic_loss"].append(loss_dict.get("critic_loss", 0))
                metrics["reward"].append(loss_dict.get("mean_reward", 0))
                metrics["ep_return"].append(np.mean(recent_returns) if recent_returns else 0)
                metrics["success_rate"].append(np.mean(recent_success) if recent_success else 0)
                metrics["env_steps"].append(total_steps)

    # ── Final evaluation across fixed difficulty levels
    print("\n  Running final evaluation (50 episodes per difficulty)...")
    eval_results = evaluate(agent, env, n_episodes=50, device=device)
    baseline_results = evaluate_pd_baseline(env, n_episodes=50)

    if plot:
        _plot_training(metrics, eval_results, pdg_cfg, baseline_results)

    return agent, metrics, {"agent": eval_results, "pd_baseline": baseline_results}


def evaluate(agent: DreamerV3, env: MarsPDGEnv,
             n_episodes: int = 50, device: str = "cpu",
             difficulties: Tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)) -> dict:
    """Evaluate trained agent across fixed difficulty levels."""
    cfg = agent.cfg
    all_results = {"summaries": {}}

    for difficulty in difficulties:
        results = []
        trajectories = []

        for ep in range(n_episodes):
            obs = env.reset(seed=5000 + ep, difficulty=difficulty)
            h = torch.zeros(1, cfg.deter_dim, device=device)
            z = torch.zeros(1, cfg.stoch_dim * cfg.stoch_classes, device=device)
            prev_action = np.zeros(cfg.action_dim, dtype=np.float32)
            done = False; ep_return = 0.0; traj = [obs.copy()]

            while not done:
                action, h, z = agent.act(obs, h, z, prev_action=prev_action,
                                         sample=False)  # deterministic
                obs, r, done, info = env.step(action)
                prev_action = action
                ep_return += r; traj.append(obs.copy())

            trajectories.append(np.array(traj))
            results.append({
                "success": info.get("success_relaxed", info.get("success", False)),
                "success_relaxed": info.get("success_relaxed", False),
                "success_strict": info.get("success_strict", False),
                "return": ep_return,
                "pos_err": info.get("pos_err", None),
                "vel_err": info.get("vel_err", None),
                "x_err": info.get("x_err", None),
                "y_err": info.get("y_err", None),
            })

        n_relaxed = sum(r["success_relaxed"] for r in results)
        n_strict = sum(r["success_strict"] for r in results)
        pos_errs = [r["pos_err"] for r in results if r["pos_err"] is not None]
        if pos_errs:
            print(
                f"  diff={difficulty:.1f}  relaxed={n_relaxed}/{n_episodes} "
                f"({100*n_relaxed/n_episodes:.0f}%)  strict={n_strict}/{n_episodes} "
                f"({100*n_strict/n_episodes:.0f}%)  Avg pos error: {np.mean(pos_errs):.1f}m"
            )
        all_results["summaries"][difficulty] = {
            "relaxed_rate": n_relaxed / n_episodes,
            "strict_rate": n_strict / n_episodes,
            "avg_pos_err": float(np.mean(pos_errs)) if pos_errs else None,
        }
        all_results[difficulty] = {"results": results, "trajectories": trajectories}

    return all_results


def evaluate_pd_baseline(env: MarsPDGEnv, n_episodes: int = 50,
                         difficulties: Tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)) -> dict:
    """Evaluate the hand-coded PD baseline for comparison."""
    all_results = {"summaries": {}}
    cfg = env.cfg

    for difficulty in difficulties:
        results = []
        trajectories = []

        for ep in range(n_episodes):
            obs = env.reset(seed=9000 + ep, difficulty=difficulty)
            done = False
            ep_return = 0.0
            traj = [obs.copy()]

            while not done:
                action = pd_guidance_action(obs, cfg)
                obs, r, done, info = env.step(action)
                ep_return += r
                traj.append(obs.copy())

            trajectories.append(np.array(traj))
            results.append({
                "success": info.get("success_relaxed", info.get("success", False)),
                "success_relaxed": info.get("success_relaxed", False),
                "success_strict": info.get("success_strict", False),
                "return": ep_return,
                "pos_err": info.get("pos_err", None),
                "vel_err": info.get("vel_err", None),
            })

        n_relaxed = sum(r["success_relaxed"] for r in results)
        n_strict = sum(r["success_strict"] for r in results)
        pos_errs = [r["pos_err"] for r in results if r["pos_err"] is not None]
        print(
            f"  PD diff={difficulty:.1f}  relaxed={n_relaxed}/{n_episodes} "
            f"({100*n_relaxed/n_episodes:.0f}%)  strict={n_strict}/{n_episodes} "
            f"({100*n_strict/n_episodes:.0f}%)"
            + (f"  Avg pos error: {np.mean(pos_errs):.1f}m" if pos_errs else "")
        )
        all_results["summaries"][difficulty] = {
            "relaxed_rate": n_relaxed / n_episodes,
            "strict_rate": n_strict / n_episodes,
            "avg_pos_err": float(np.mean(pos_errs)) if pos_errs else None,
        }
        all_results[difficulty] = {"results": results, "trajectories": trajectories}

    return all_results


def _plot_training(metrics: dict, eval_results: dict, pdg_cfg: PDGConfig,
                   baseline_results: Optional[dict] = None):
    diff_keys = sorted(k for k in eval_results.keys() if isinstance(k, float))
    plot_key = max(diff_keys)
    plot_eval = eval_results[plot_key]
    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    def styled_ax(pos):
        ax = fig.add_subplot(pos, facecolor="#161b22")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        return ax

    steps = metrics["env_steps"]

    # Row 1: Training curves
    ax1 = styled_ax(gs[0, 0])
    ax1.plot(steps, metrics["wm_loss"], color="#00d4ff", lw=1.5)
    ax1.set_title("World Model Loss", color="white", fontsize=9)
    ax1.set_xlabel("Env Steps"); ax1.set_ylabel("Loss")

    ax2 = styled_ax(gs[0, 1])
    ax2.plot(steps, metrics["actor_loss"],  color="#ff6b6b", lw=1.5, label="Actor")
    ax2.plot(steps, metrics["critic_loss"], color="#ffd93d", lw=1.5, label="Critic")
    ax2.set_title("Actor / Critic Loss", color="white", fontsize=9)
    ax2.legend(fontsize=7, facecolor="#161b22", labelcolor="white", edgecolor="#30363d")

    ax3 = styled_ax(gs[0, 2])
    ax3.plot(steps, metrics["ep_return"], color="#00ff88", lw=1.5)
    ax3.set_title("Episode Return", color="white", fontsize=9)
    ax3.set_xlabel("Env Steps"); ax3.set_ylabel("Return")

    ax4 = styled_ax(gs[0, 3])
    ax4.plot(steps, [s*100 for s in metrics["success_rate"]], color="#c77dff", lw=1.5)
    ax4.set_title("Success Rate (%)", color="white", fontsize=9)
    ax4.axhline(100*pdg_cfg.pos_tol/pdg_cfg.alt_init, color="#ff6b6b",
                lw=0.8, ls="--", alpha=0.5)
    ax4.set_ylim(0, 100)

    # Row 2: Evaluation results
    trajectories = plot_eval["trajectories"]
    results      = plot_eval["results"]

    ax5 = styled_ax(gs[1, 0:2])
    for traj, res in zip(trajectories[:15], results[:15]):
        horiz = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)   # horizontal radius
        alt   = traj[:, 2]   # z (altitude)
        color = "#00ff88" if res["success"] else "#ff6b6b"
        ax5.plot(horiz, alt, color=color, alpha=0.6, lw=1.2)
    ax5.axhline(0, color="white", lw=0.5, ls="--")
    ax5.set_title("Descent Trajectories  (horizontal radius vs altitude)",
                   color="white", fontsize=9)
    ax5.set_xlabel("Horizontal Radius (m)"); ax5.set_ylabel("Altitude (m)")

    ax6 = styled_ax(gs[1, 2])
    for res, traj in zip(results, trajectories):
        fx = traj[-1, 0]
        fy = traj[-1, 1]
        ax6.scatter(fx, fy, color="#00ff88" if res["success"] else "#ff6b6b",
                    s=25, alpha=0.7, zorder=3)
    ax6.axvline(0, color="white", lw=0.8, ls="--")
    ax6.axhline(0, color="white", lw=0.8, ls="--")
    landing_circle = plt.Circle((0, 0), pdg_cfg.pos_tol, color="#ffff00",
                                alpha=0.12, label=f"{pdg_cfg.pos_tol}m radius")
    ax6.add_patch(landing_circle)
    ax6.set_title("Landing Scatter (x-y)", color="white", fontsize=9)
    ax6.set_xlabel("Final x (m)"); ax6.set_ylabel("Final y (m)")
    ax6.set_aspect("equal", adjustable="box")
    ax6.legend(fontsize=7, facecolor="#161b22", labelcolor="white", edgecolor="#30363d")

    ax7 = styled_ax(gs[1, 3])
    pos_errs = [r["pos_err"] for r in results if r["pos_err"] is not None]
    vel_errs = [r["vel_err"] for r in results if r["vel_err"] is not None]
    if pos_errs:
        ax7.hist(pos_errs, bins=20, color="#00d4ff", alpha=0.7, label="pos err (m)")
        ax7.axvline(pdg_cfg.pos_tol, color="#ffff00", lw=1.5, ls="--",
                    label=f"{pdg_cfg.pos_tol}m target")
    ax7.set_title("Landing Error Distribution", color="white", fontsize=9)
    ax7.legend(fontsize=7, facecolor="#161b22", labelcolor="white", edgecolor="#30363d")

    n_succ = sum(r["success_relaxed"] for r in results)
    n_strict = sum(r["success_strict"] for r in results)
    summary_text = "  ".join(
        f"d={d:.1f}: {eval_results['summaries'][d]['relaxed_rate']*100:.0f}%/"
        f"{eval_results['summaries'][d]['strict_rate']*100:.0f}%"
        for d in diff_keys
    )
    baseline_text = ""
    if baseline_results is not None:
        baseline_text = "  |  PD " + "  ".join(
            f"d={d:.1f}: {baseline_results['summaries'][d]['relaxed_rate']*100:.0f}%/"
            f"{baseline_results['summaries'][d]['strict_rate']*100:.0f}%"
            for d in diff_keys
        )
    plt.suptitle(
        f"DreamerV3 · Mars PDG  |  Eval diff={plot_key:.1f}  |  "
        f"Relaxed: {n_succ}/{len(results)} ({100*n_succ/len(results):.0f}%)  "
        f"Strict: {n_strict}/{len(results)} ({100*n_strict/len(results):.0f}%)  |  "
        f"{summary_text}{baseline_text}",
        color="white", fontsize=11, fontweight="bold")

    plt.savefig("dreamerv3_pdg_output.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Saved → dreamerv3_pdg_output.png")
    plt.close(fig)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    agent, metrics, eval_results = train_dreamerv3(
        n_env_steps   = 300_000,
        train_every   = 5,
        seed_episodes = 20,
        device        = None,   # auto-detects GPU
        plot          = True,
    )

    torch.save({"agent": agent.state_dict(), "metrics": metrics},
               "dreamerv3_final.pth")
    print("Saved → dreamerv3_final.pth")
