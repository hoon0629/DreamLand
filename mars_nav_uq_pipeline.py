"""
Mars Autonomous Navigation Pipeline — Uncertainty-Aware
=========================================================
AI4MARS + DeepLabV3+ + lightning-uq-box (MC Dropout UQ) + A* path planning

Key upgrade over v1: terrain cost map is now modulated by per-pixel
UNCERTAINTY from MC Dropout. Uncertain terrain = elevated traversal cost.
This makes the planner risk-aware, not just obstacle-aware.

Cost formula:
    cost(pixel) = base_terrain_cost × (1 + α × epistemic_uncertainty)

Install:
    pip install torch torchvision lightning-uq-box numpy pillow
    pip install matplotlib networkx requests scipy lightning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from PIL import Image
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

# lightning-uq-box imports
# MC Dropout wraps any model and runs N stochastic forward passes
from lightning_uq_box.models import MCDropoutSegmentation


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

AI4MARS_CLASSES = {
    0: "soil",
    1: "bedrock",
    2: "sand",
    3: "rock",
}

# Base traversability cost per class
BASE_COST = {
    0: 1.0,    # soil    → easy
    1: 3.0,    # bedrock → moderate
    2: 7.0,    # sand    → slippage risk
    3: 100.0,  # rock    → obstacle
}

CLASS_COLORS = {
    0: (194, 178, 128),  # soil    → sandy beige
    1: (128, 128, 128),  # bedrock → gray
    2: (210, 180, 140),  # sand    → tan
    3: (139, 69, 19),    # rock    → brown
}

# How strongly uncertainty inflates the cost
# α=5 means max-uncertainty terrain gets 6x base cost
UNCERTAINTY_ALPHA = 5.0


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_image(source: str) -> np.ndarray:
    """Load from file path or URL → HxWx3 uint8 array."""
    if source.startswith("http"):
        req = urllib.request.Request(source, headers={"User-Agent": "MarsNavBot/1.0"})
        with urllib.request.urlopen(req) as resp:
            img = Image.open(resp).convert("RGB")
    else:
        img = Image.open(source).convert("RGB")
    return np.array(img)


def fetch_nasa_rover_image(api_key: str = "DEMO_KEY",
                            sol: int = 1000,
                            rover: str = "curiosity",
                            camera: str = "NAVCAM") -> str:
    """Fetch a rover image URL from NASA Mars Rover Photos API."""
    import json
    url = (
        f"https://api.nasa.gov/mars-photos/api/v1/rovers/{rover}/photos"
        f"?sol={sol}&camera={camera}&api_key={api_key}&page=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "MarsNavBot/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    photos = data.get("photos", [])
    if not photos:
        raise ValueError(f"No photos for sol={sol}, camera={camera}")
    img_url = photos[0]["img_src"]
    print(f"  Found {len(photos)} photos → {img_url}")
    return img_url


# ─────────────────────────────────────────────
# SEGMENTATION MODEL
# ─────────────────────────────────────────────

def _build_base_deeplabv3(num_classes: int = 4,
                            dropout_p: float = 0.3) -> nn.Module:
    """
    Build DeepLabV3+ with dropout injected into the ASPP and classifier.
    Dropout enables MC Dropout UQ: during inference, run N forward passes
    with dropout active to sample from the approximate posterior.
    """
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)

    # Inject dropout into the ASPP module (before classifier)
    model.classifier = nn.Sequential(
        model.classifier[0],                        # ASPP
        nn.Dropout2d(p=dropout_p),                  # ← dropout for UQ
        nn.Conv2d(256, num_classes, kernel_size=1), # output head (4 classes)
    )
    model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ─────────────────────────────────────────────
# UNCERTAINTY-AWARE SEGMENTER
# ─────────────────────────────────────────────

class UncertaintyAwareSegmenter:
    """
    DeepLabV3+ wrapped with MC Dropout via lightning-uq-box.

    For each image, runs `num_mc_samples` stochastic forward passes
    (dropout active) and returns:
      - mean_pred   : (H, W)   — most likely terrain class
      - uncertainty : (H, W)   — epistemic uncertainty (0=certain, 1=max)

    Uncertainty is computed as predictive entropy over the MC samples:
        H = -Σ p̄_c · log(p̄_c)    where p̄_c = mean softmax prob of class c
    Normalized to [0, 1] by dividing by log(num_classes).

    To use your fine-tuned AI4MARS weights:
        ckpt = torch.load("checkpoints/deeplabv3_ai4mars_best.pth")
        self.model.base_model.load_state_dict(ckpt["model"])
    """

    def __init__(self,
                 num_mc_samples: int = 20,
                 dropout_p: float = 0.3,
                 num_classes: int = 4,
                 device: str = None,
                 checkpoint_path: str = None):
        self.num_classes = num_classes
        self.num_mc_samples = num_mc_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  Building DeepLabV3+ with MC Dropout (p={dropout_p}, "
              f"samples={num_mc_samples}) on {self.device}...")

        base_model = _build_base_deeplabv3(num_classes=num_classes,
                                            dropout_p=dropout_p)

        # Load fine-tuned AI4MARS weights if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            base_model.load_state_dict(ckpt["model"])
            print(f"  Loaded checkpoint: {checkpoint_path} "
                  f"(mIoU={ckpt.get('miou', '?'):.4f})")
        else:
            print("  Using ImageNet pretrained weights (no AI4MARS fine-tune).")
            print("  Run train_ai4mars.py to get true Mars terrain accuracy.")

        # Wrap with MCDropoutSegmentation from lightning-uq-box
        # This handles enabling dropout at inference + multi-pass aggregation
        self.model = MCDropoutSegmentation(
            model=base_model,
            num_mc_samples=num_mc_samples,
        ).to(self.device)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        print("  Model ready.")

    def predict(self, image: np.ndarray) -> dict:
        """
        Run uncertainty-aware segmentation.

        Args:
            image: HxWx3 uint8 numpy array

        Returns dict with:
            "labels"      : (H, W) int — predicted terrain class per pixel
            "probs"       : (C, H, W) float — mean softmax probabilities
            "uncertainty" : (H, W) float in [0,1] — normalized predictive entropy
            "std"         : (H, W) float — std of softmax prob across MC samples
        """
        pil_img = Image.fromarray(image)
        inp = self.transform(pil_img).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        self.model.eval()
        with torch.no_grad():
            # MCDropoutSegmentation runs num_mc_samples forward passes internally
            # Returns dict with keys: "logits", "probs", "pred" depending on version
            out = self.model(inp)

        # Handle different lightning-uq-box output formats
        if isinstance(out, dict):
            if "probs" in out:
                # Shape: (num_mc_samples, 1, C, H, W) or (1, C, H, W)
                probs = out["probs"].squeeze()
            elif "logits" in out:
                probs = torch.softmax(out["logits"].squeeze(), dim=0)
            else:
                probs = list(out.values())[0].squeeze()
        else:
            probs = torch.softmax(out.squeeze(), dim=0)

        # If we got MC samples stacked: (num_mc_samples, C, H, W)
        if probs.dim() == 4:
            mean_probs = probs.mean(0)          # (C, H, W)
            std_probs  = probs.std(0).mean(0)   # (H, W)
        else:
            mean_probs = probs                  # (C, H, W)
            std_probs  = torch.zeros(probs.shape[1:])

        mean_probs_np = mean_probs.cpu().numpy()   # (C, H, W)
        std_np = std_probs.cpu().numpy()            # (H, W)

        # Predictive entropy as epistemic uncertainty metric
        # H(ȳ) = -Σ_c p̄_c log(p̄_c)
        eps = 1e-8
        entropy = -np.sum(mean_probs_np * np.log(mean_probs_np + eps), axis=0)
        # Normalize by log(C) so uncertainty ∈ [0, 1]
        max_entropy = np.log(self.num_classes)
        uncertainty = entropy / max_entropy  # (H, W) in [0, 1]

        labels = mean_probs_np.argmax(axis=0).astype(np.uint8)  # (H, W)

        return {
            "labels":      labels,
            "probs":       mean_probs_np,
            "uncertainty": uncertainty,
            "std":         std_np,
        }

    @staticmethod
    def _fallback_segment(image: np.ndarray, num_classes: int = 4) -> dict:
        """
        Fallback proxy segmentation (no fine-tuning) using color statistics.
        Returns same dict structure as predict().
        """
        # Simple heuristic: dark pixels → rocks, bright reddish → soil/sand
        h, w = image.shape[:2]
        brightness = image.mean(axis=2) / 255.0
        redness    = (image[:,:,0].astype(float) - image[:,:,2]) / 255.0

        labels = np.zeros((h, w), dtype=np.uint8)
        labels[brightness < 0.25]  = 3  # dark → rock
        labels[brightness > 0.75]  = 2  # very bright → sand
        labels[(redness > 0.2) & (labels == 0)] = 0  # reddish → soil
        labels[labels == 0] = 1  # default → bedrock

        # Simulate uncertainty: edges and mid-brightness zones are uncertain
        edge = np.abs(np.gradient(brightness.astype(float))[0]) + \
               np.abs(np.gradient(brightness.astype(float))[1])
        uncertainty = np.clip(edge * 3, 0, 1)

        # Build mean_probs from one-hot + uncertainty smearing
        probs = np.zeros((num_classes, h, w), dtype=np.float32)
        for c in range(num_classes):
            mask = (labels == c).astype(float)
            # Add uncertainty mass to adjacent classes
            probs[c] = mask * (1 - uncertainty * 0.6)
        # Renormalize
        probs_sum = probs.sum(axis=0, keepdims=True) + 1e-8
        probs /= probs_sum

        return {"labels": labels, "probs": probs,
                "uncertainty": uncertainty, "std": uncertainty * 0.5}


# ─────────────────────────────────────────────
# UNCERTAINTY-AWARE COST MAP
# ─────────────────────────────────────────────

def build_uncertainty_cost_map(labels: np.ndarray,
                                uncertainty: np.ndarray,
                                alpha: float = UNCERTAINTY_ALPHA) -> np.ndarray:
    """
    Combine terrain class cost with epistemic uncertainty.

    cost(x,y) = base_terrain_cost(x,y) × (1 + α × uncertainty(x,y))

    Effect:
    - Certain soil (u≈0)  → cost = 1.0  (route freely)
    - Uncertain soil (u≈1) → cost = 6.0  (avoid: might actually be sand/rock)
    - Certain rock (u≈0)  → cost = 100   (hard obstacle)
    - Uncertain rock      → cost = 600   (extra dangerous)

    Args:
        labels:      (H, W) terrain class indices
        uncertainty: (H, W) normalized entropy in [0, 1]
        alpha:       uncertainty penalty strength

    Returns:
        cost_map: (H, W) float32
    """
    base_cost = np.zeros(labels.shape, dtype=np.float32)
    for cls, cost in BASE_COST.items():
        base_cost[labels == cls] = cost

    uncertainty_multiplier = 1.0 + alpha * uncertainty
    cost_map = base_cost * uncertainty_multiplier
    return cost_map.astype(np.float32)


def downsample_cost_map(cost_map: np.ndarray,
                         target_size: int = 100) -> np.ndarray:
    """Downsample cost map for planning grid (preserves obstacle signal)."""
    from scipy.ndimage import zoom
    h, w = cost_map.shape
    scale = target_size / max(h, w)
    return zoom(cost_map, scale, order=1)


# ─────────────────────────────────────────────
# A* PATH PLANNER (same as v1)
# ─────────────────────────────────────────────

def build_navigation_graph(cost_map: np.ndarray,
                            obstacle_threshold: float = 50.0) -> nx.DiGraph:
    h, w = cost_map.shape
    G = nx.DiGraph()
    for r in range(h):
        for c in range(w):
            if cost_map[r, c] < obstacle_threshold:
                G.add_node((r, c))
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for r in range(h):
        for c in range(w):
            if (r, c) not in G:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in G:
                    dist = 1.414 if (dr != 0 and dc != 0) else 1.0
                    G.add_edge((r, c), (nr, nc),
                               weight=dist * cost_map[nr, nc])
    return G


def heuristic(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def plan_path(cost_map: np.ndarray,
              start: tuple, goal: tuple) -> list:
    print(f"  Building graph ({cost_map.shape[0]}×{cost_map.shape[1]})...")
    G = build_navigation_graph(cost_map)
    start = _snap_to_free(cost_map, start)
    goal  = _snap_to_free(cost_map, goal)
    print(f"  A*: {start} → {goal}")
    try:
        path = nx.astar_path(G, start, goal, heuristic=heuristic, weight="weight")
        cost = nx.astar_path_length(G, start, goal, heuristic=heuristic, weight="weight")
        print(f"  Path: {len(path)} waypoints, total cost={cost:.2f}")
        return path
    except nx.NetworkXNoPath:
        print("  No path found.")
        return []


def _snap_to_free(cost_map, point, threshold=50.0):
    r, c = point
    h, w = cost_map.shape
    if cost_map[r, c] < threshold:
        return point
    for rad in range(1, max(h, w)):
        for dr in range(-rad, rad+1):
            for dc in range(-rad, rad+1):
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w:
                    if cost_map[nr, nc] < threshold:
                        return (nr, nc)
    return point


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def visualize(original_img, labels, uncertainty, cost_map, path,
              start, goal, save_path=None):
    """5-panel visualization: image | segmentation | uncertainty | cost | path"""
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="white")

    # 1. Original
    axes[0].imshow(original_img)
    axes[0].set_title("Mars Rover Image", color="white", fontsize=10, pad=8)
    axes[0].axis("off")

    # 2. Segmentation
    h, w = labels.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        seg_rgb[labels == cls] = color
    img_r = np.array(Image.fromarray(original_img).resize((w, h)))
    blend = np.clip(0.45 * img_r / 255 + 0.55 * seg_rgb / 255, 0, 1)
    axes[1].imshow(blend)
    axes[1].set_title("Terrain Segmentation\n(DeepLabV3+)", color="white", fontsize=10, pad=8)
    axes[1].axis("off")
    patches = [mpatches.Patch(color=np.array(c)/255, label=AI4MARS_CLASSES[i])
               for i, c in CLASS_COLORS.items()]
    axes[1].legend(handles=patches, loc="lower right", fontsize=7,
                   facecolor="#161b22", labelcolor="white", edgecolor="#30363d")

    # 3. Uncertainty map (key addition)
    im_u = axes[2].imshow(uncertainty, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Epistemic Uncertainty\n(MC Dropout — lightning-uq-box)",
                       color="white", fontsize=10, pad=8)
    axes[2].axis("off")
    cbar_u = plt.colorbar(im_u, ax=axes[2], fraction=0.046, pad=0.04)
    cbar_u.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar_u.ax.yaxis.get_ticklabels(), color="white")
    cbar_u.set_label("0=certain  1=uncertain", color="white", fontsize=7)

    # 4. Uncertainty-aware cost map
    im_c = axes[3].imshow(cost_map, cmap="RdYlGn_r", vmin=1, vmax=30)
    axes[3].set_title("Uncertainty-Aware Cost Map\ncost = base × (1 + α·uncertainty)",
                       color="white", fontsize=10, pad=8)
    axes[3].axis("off")
    cbar_c = plt.colorbar(im_c, ax=axes[3], fraction=0.046, pad=0.04)
    cbar_c.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar_c.ax.yaxis.get_ticklabels(), color="white")

    # 5. A* path on cost map
    axes[4].imshow(cost_map, cmap="RdYlGn_r", vmin=1, vmax=30)
    if path:
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        axes[4].plot(pc, pr, color="#00d4ff", linewidth=2.5, label="A* path", zorder=3)
        axes[4].plot(pc[0], pr[0], "o", color="#00ff88", markersize=10,
                     label="Start", zorder=4)
        axes[4].plot(pc[-1], pr[-1], "*", color="#ff6b6b", markersize=14,
                     label="Goal", zorder=4)
    axes[4].set_title(f"Risk-Aware A* Path\n({len(path)} waypoints)",
                       color="white", fontsize=10, pad=8)
    axes[4].axis("off")
    axes[4].legend(loc="lower right", fontsize=7, facecolor="#161b22",
                   labelcolor="white", edgecolor="#30363d")

    plt.suptitle(
        "Mars Autonomous Navigation  |  DeepLabV3+ · MC Dropout (lightning-uq-box) · A*",
        color="white", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(image_source: str = None,
                 api_key: str = "DEMO_KEY",
                 sol: int = 1000,
                 num_mc_samples: int = 20,
                 dropout_p: float = 0.3,
                 uncertainty_alpha: float = UNCERTAINTY_ALPHA,
                 planning_grid_size: int = 80,
                 checkpoint_path: str = None,
                 output_path: str = "mars_nav_uq_output.png"):
    """
    Full uncertainty-aware pipeline.

    Args:
        image_source:      Local path or URL. None → fetch from NASA API.
        api_key:           NASA API key (free at api.nasa.gov).
        sol:               Martian day for NASA API fetch.
        num_mc_samples:    MC Dropout passes (more = better UQ, slower).
        dropout_p:         Dropout probability for MC Dropout.
        uncertainty_alpha: How strongly uncertainty inflates traversal cost.
        planning_grid_size: Resolution of planning grid.
        checkpoint_path:   Path to fine-tuned AI4MARS weights (.pth).
        output_path:       Where to save visualization.
    """
    print("\n🚀 Mars Navigation Pipeline (Uncertainty-Aware)")
    print("=" * 55)

    # 1. Load image
    print("\n[1/5] Loading Mars image...")
    if image_source is None:
        try:
            url = fetch_nasa_rover_image(api_key=api_key, sol=sol)
            image = load_image(url)
        except Exception as e:
            print(f"  API failed ({e}). Using synthetic image.")
            image = _synthetic_mars(256)
    else:
        image = load_image(image_source)
    print(f"  Shape: {image.shape}")

    # 2. Uncertainty-aware segmentation
    print(f"\n[2/5] Running MC Dropout segmentation "
          f"({num_mc_samples} samples)...")
    try:
        segmenter = UncertaintyAwareSegmenter(
            num_mc_samples=num_mc_samples,
            dropout_p=dropout_p,
            checkpoint_path=checkpoint_path,
        )
        result = segmenter.predict(image)
    except ImportError:
        print("  lightning-uq-box not found. Using fallback segmenter.")
        print("  Install: pip install lightning-uq-box")
        result = UncertaintyAwareSegmenter._fallback_segment(image)

    labels      = result["labels"]       # (H, W)
    uncertainty = result["uncertainty"]  # (H, W) in [0, 1]

    # Print per-class stats
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {AI4MARS_CLASSES[cls]:10s}: {100*cnt/labels.size:5.1f}%")
    print(f"  Avg uncertainty: {uncertainty.mean():.3f}  "
          f"Max: {uncertainty.max():.3f}")

    # 3. Build uncertainty-aware cost map
    print(f"\n[3/5] Building uncertainty-aware cost map (α={uncertainty_alpha})...")
    cost_map_full = build_uncertainty_cost_map(labels, uncertainty, uncertainty_alpha)
    cost_map = downsample_cost_map(cost_map_full, target_size=planning_grid_size)
    unc_down = downsample_cost_map(uncertainty, target_size=planning_grid_size)
    print(f"  Cost range: [{cost_map.min():.1f}, {cost_map.max():.1f}]")

    # 4. A* planning
    print("\n[4/5] Planning path...")
    h, w = cost_map.shape
    start, goal = (5, 5), (h-6, w-6)
    path = plan_path(cost_map, start, goal)

    # 5. Visualize
    print("\n[5/5] Visualizing...")
    img_disp = np.array(Image.fromarray(image).resize((labels.shape[1], labels.shape[0])))
    visualize(img_disp, labels, uncertainty, cost_map, path,
              start, goal, save_path=output_path)

    # Summary
    print("\n✅ Done!")
    if path:
        path_costs = [cost_map[r, c] for r, c in path]
        path_unc   = [unc_down[r, c] for r, c in path]
        print(f"  Waypoints      : {len(path)}")
        print(f"  Avg path cost  : {np.mean(path_costs):.2f}")
        print(f"  Avg path uncertainty: {np.mean(path_unc):.3f} "
              f"(lower = more confident route)")
    return {"labels": labels, "uncertainty": uncertainty,
            "cost_map": cost_map, "path": path}


def _synthetic_mars(size=256) -> np.ndarray:
    np.random.seed(0)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:,:,0] = np.clip(np.random.normal(180, 30, (size,size)), 100, 255)
    img[:,:,1] = np.clip(np.random.normal(100, 20, (size,size)), 50,  180)
    img[:,:,2] = np.clip(np.random.normal(60,  15, (size,size)), 20,  120)
    for _ in range(20):
        r = np.random.randint(10, size-20)
        c = np.random.randint(10, size-20)
        rr, cc = np.random.randint(5, 25), np.random.randint(5, 25)
        img[r:r+rr, c:c+cc] = [80, 50, 30]
    return img


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    USAGE:

    1. Quick start (offline, synthetic image):
       python mars_nav_uq_pipeline.py

    2. NASA live imagery:
       results = run_pipeline(api_key="YOUR_KEY", sol=1000)

    3. Local AI4MARS image + fine-tuned weights:
       results = run_pipeline(
           image_source="path/to/ai4mars_image.jpg",
           checkpoint_path="checkpoints/deeplabv3_ai4mars_best.pth",
           num_mc_samples=30,   # more samples → better uncertainty estimate
       )

    4. Tune uncertainty sensitivity:
       run_pipeline(uncertainty_alpha=10.0)  # very risk-averse planner
       run_pipeline(uncertainty_alpha=1.0)   # mild penalty for uncertainty
    """
    results = run_pipeline(
        api_key="DEMO_KEY",
        sol=1000,
        num_mc_samples=20,
        dropout_p=0.3,
        uncertainty_alpha=5.0,
        planning_grid_size=80,
        output_path="mars_nav_uq_output.png",
    )
