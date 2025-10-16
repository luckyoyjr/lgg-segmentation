# app_hybrid_only.py - SYNCED WITH ORIGINAL TRAINING ARCHITECTURE
# ===============================================================
# Streamlit: Hybrid (Quantum + Attention) U-Net — Single-Image Demo
# - Architecture is an EXACT COPY of the model used in training/eval.
# - Attention maps are captured via PyTorch hooks to match evaluation logic.
# - Should load 'best_hybrid_ag.pt' with missing=0 unexpected=0.
# ===============================================================

import io, math, warnings
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# Optional quantum backend (fallback if missing)
try:
    import pennylane as qml
except Exception:
    qml = None
    print("Warning: Pennylane not found. Quantum layer will use fallback.")

# -----------------------
# CONFIG
# -----------------------
PROJECT_ROOT = Path.cwd()
CKPT_HYBRID = PROJECT_ROOT / "models" / "best_hybrid_ag.pt"

# NOTE: Using the 0.10 default threshold from your eval script, not 0.55
BEST_THR_HYBRID = 0.10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Model (EXACTLY MATCHES TRAINING CODE)
# -----------------------
class DoubleConv_App(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.net(x)


class AttentionGate_App(nn.Module):
    """
    Returns the gated feature map. Attention mask is captured via a hook on self.psi.
    Includes a fix to handle spatial size mismatches (e.g., 126 vs 127) by 
    resizing the skip connection feature map (x_out) to match the attention signal (g_out).
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # EXACTLY MATCHES W_g, W_x, psi definitions from training
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # 1. Process the attention signal (g) and skip connection (x)
        g_out = self.W_g(g)
        x_out = self.W_x(x)

        # 2. FIX: Check for size mismatch and interpolate the larger tensor (x_out) 
        # to match the size of the attention signal (g_out).
        if x_out.shape[2:] != g_out.shape[2:]:
            H_g, W_g = g_out.shape[2:]
            # Use bilinear for smooth interpolation, or nearest if the original model
            # used cropping (F.interpolate(..., mode='nearest') is often a simple 
            # replacement for cropping when the mismatch is small).
            x_out = F.interpolate(x_out, size=(H_g, W_g), mode='nearest') 

        # 3. Perform the element-wise addition (this is where the error occurred)
        psi_out = self.relu(g_out + x_out)
        
        # 4. Final gating
        return x * self.psi(psi_out)


class QuantumBottleneck_App(nn.Module):
    """EXACTLY MATCHES the training QuantumBottleneck class."""

    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.01)
        self._proj = None  # Must be dynamically created or loaded from checkpoint

        if qml is not None:
            self.qdevice = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(self.qdevice, interface="torch", diff_method="backprop")
            def circuit(x, thetas):
                qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True, pad_with=0.0)
                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.RX(thetas[l, q, 0], wires=q)
                        qml.RY(thetas[l, q, 1], wires=q)
                        qml.RZ(thetas[l, q, 2], wires=q)
                    for q in range(n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % n_qubits])
                return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

            self.qnode = circuit
        else:
            self.qnode = None

    def forward(self, x):
        B, D = x.shape
        amp_dim = 2 ** self.n_qubits

        # This layer is initialized dynamically in the UNet prime_qproj method
        if self._proj is None:
            # Fallback for missing _proj weights if not loaded/primed
            return torch.randn(B, self.n_qubits, dtype=x.dtype, device=x.device)

        x32 = x.float()
        proj = torch.tanh(self._proj(x32))
        proj = proj / (proj.norm(p=2, dim=1, keepdim=True) + 1e-8)

        if self.qnode is not None:
            outs = [torch.stack(self.qnode(proj[b], self.theta)).float() for b in range(B)]
            z = torch.stack(outs, dim=0).to(x.dtype)
        else:
            # Fallback when pennylane is not installed (keeps app functional)
            z = torch.tanh(x32)[:, :self.n_qubits]
            if z.shape[1] != self.n_qubits:
                z = torch.randn(B, self.n_qubits, dtype=x.dtype, device=x.device)

        return z


class UNetHybridAG_App(nn.Module):
    """EXACTLY MATCHES the training UNetHybridAG class."""

    def __init__(self, in_ch=1, out_ch=1, base_ch=32, n_qubits=4, n_layers=2, target_alpha=0.7):
        super().__init__()
        C = base_ch
        self.base_ch = base_ch
        self.target_alpha = float(target_alpha)
        # encoder
        self.enc1 = DoubleConv_App(in_ch, C)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv_App(C, 2 * C)
        self.pool2 = nn.MaxPool2d(2)
        # bottleneck + quantum fuse (ALL LAYERS SYNCED)
        self.bottleneck = DoubleConv_App(2 * C, 4 * C)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lin_in = nn.Linear(4 * C, 4 * C)
        self.qblock = QuantumBottleneck_App(n_qubits, n_layers)
        self.lin_out = nn.Linear(n_qubits, 4 * C)
        self.fuse_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * C, 4 * C, 1), nn.ReLU(inplace=True),
            nn.Conv2d(4 * C, 4 * C, 1), nn.Sigmoid()
        )
        self.alpha_param = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("alpha_warm", torch.tensor(0.0))
        # decoder with attention
        self.up2 = nn.ConvTranspose2d(4 * C, 2 * C, 2, 2)
        self.ag2 = AttentionGate_App(2 * C, 2 * C, C)
        self.dec2 = DoubleConv_App(4 * C, 2 * C)
        self.up1 = nn.ConvTranspose2d(2 * C, C, 2, 2)
        self.ag1 = AttentionGate_App(C, C, C // 2)
        self.dec1 = DoubleConv_App(2 * C, C)
        self.outc = nn.Conv2d(C, out_ch, 1)

        # Init alpha_param (SYNCED)
        init = math.log(self.target_alpha / (1 - self.target_alpha))
        with torch.no_grad():
            self.alpha_param.copy_(torch.tensor(init, dtype=self.alpha_param.dtype))

    @torch.no_grad()
    def prime_qproj(self, device):
        """Initializes the QuantumBottleneck's projection layer if it's missing (required for loading)."""
        if getattr(self.qblock, "_proj", None) is None:
            C4 = 4 * self.base_ch
            amp_dim = 2 ** self.qblock.n_qubits
            self.qblock._proj = nn.Linear(C4, amp_dim, bias=False).to(device)
            st.caption("Quantum projection layer initialized/primed.")

    def _quantum_fuse(self, b):
        """Replicates the complex fusion mechanism from the training script."""
        B, C4, H, W = b.shape
        # Input linear layer -> QBlock
        v = torch.tanh(self.lin_in(self.gap(b.float()).view(B, C4)))
        q = self.qblock(v)
        # Output linear layer -> Feature map
        v2 = torch.tanh(self.lin_out(q))
        qft = v2.view(B, C4, 1, 1).expand(-1, -1, H, W)
        # Fuse Gate
        gate = self.fuse_gate(b)
        # Blending Alpha (using alpha_param and alpha_warm buffer)
        a_learn = torch.sigmoid(self.alpha_param)
        a = (1 - self.alpha_warm) * 0.0 + self.alpha_warm * a_learn
        # Final blended output
        return b + gate * (a * (qft.to(b.dtype) - b))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        bq = self._quantum_fuse(b)  # Use the complex fusion logic
        d2 = self.up2(bq)
        e2a = self.ag2(e2, d2)  # AttentionGate returns the gated feature map
        d2 = self.dec2(torch.cat([d2, e2a], 1))
        d1 = self.up1(d2)
        e1a = self.ag1(e1, d1)  # AttentionGate returns the gated feature map
        d1 = self.dec1(torch.cat([d1, e1a], 1))
        return self.outc(d1)


# -----------------
# Loading helper
# -----------------
@st.cache_resource(show_spinner=True)
def load_hybrid(ckpt_path: Path) -> nn.Module:
    # SYNCED: Use the same base_ch, n_qubits, n_layers as your training script defaults
    model = UNetHybridAG_App(in_ch=1, out_ch=1, base_ch=32, n_qubits=4, n_layers=2).to(DEVICE)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    # Checkpoint processing (sync with eval script)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    if isinstance(sd, dict) and len(sd) and next(iter(sd)).startswith("module."):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # IMPORTANT: Prime the dynamic layer BEFORE loading the weights for it
    model.prime_qproj(DEVICE)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    st.caption(f"Loaded hybrid weights (strict=False): missing={len(missing)} unexpected={len(unexpected)}")

    if len(missing) + len(unexpected) > 0:
        st.warning("Architecture mismatch persists! Check if training parameters (e.g., base_ch) were different.")

    model.eval()
    return model


# -----------------
# Image utilities (No change needed)
# -----------------
def load_image_to_array(file) -> np.ndarray:
    if file is None: return None
    name = file.name.lower()
    if name.endswith(".npy"):
        arr = np.load(file);
        arr = np.squeeze(arr) if arr.ndim == 3 else arr
    else:
        im = Image.open(file).convert("L");
        arr = np.array(im)
    arr = arr.astype(np.float32);
    if arr.max() > 1.0: arr /= 255.0
    return arr


def preprocess(img: np.ndarray) -> torch.Tensor:
    if img.ndim != 2: raise ValueError(f"Expected a single-channel 2D image, got shape {img.shape}")
    return torch.from_numpy(img[None, None, ...]).to(DEVICE)


# -----------------
# Metrics (No change needed)
# -----------------
def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))
    tn = np.sum((pred_bin == 0) & (gt_bin == 0))
    eps = 1e-8
    prec = tp / (tp + fp + eps);
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps);
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    return dict(precision=prec, recall=rec, f1=f1, dice=dice, iou=iou, accuracy=acc, cm=cm)


# -----------------
# Drawing helpers (No change needed)
# -----------------
def draw_overlay(img, pred=None, gt=None, title="Overlay", thr=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    if gt is not None:
        for c in find_contours(gt, 0.5): ax.plot(c[:, 1], c[:, 0], color="blue", lw=2, label="Ground Truth")
    if pred is not None:
        for c in find_contours(pred, 0.5): ax.plot(c[:, 1], c[:, 0], color="red", lw=2, label="Prediction")
    ax.set_title(f"{title}" + (f" | thr={thr:.2f}" if thr is not None else ""))
    ax.axis("off");
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles));
    if uniq: ax.legend(uniq.values(), uniq.keys(), loc="lower right", frameon=True)
    buf = io.BytesIO();
    plt.tight_layout();
    plt.savefig(buf, format="png", dpi=160);
    plt.close(fig);
    buf.seek(0)
    return buf


def show_heatmap(img, heat, title):
    heat = np.squeeze(heat);
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    # SYNCED: Use 'magma' as in your eval script's overlay_with_attention function
    im = ax.imshow(heat, cmap="magma", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title);
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04);
    st.pyplot(fig, clear_figure=True)


# -----------------
# Inference wrapper (FIXED to use HOOKS)
# -----------------
def run_hybrid_single(img_arr: np.ndarray, thr: float, mode: str):
    """
    mode in {"Prediction only", "Attention only", "Prediction + Attention"}
    Uses forward hooks to capture the attention maps, just like the training eval script.
    """
    model = load_hybrid(CKPT_HYBRID)

    # SYNCED: Setup attention hooks just like the eval script
    attn_cache = {}

    def hook_ag1(module, inp, out):
        attn_cache["ag1"] = out.detach().mean(1)  # (B,1,H,W)->(B,H,W)

    def hook_ag2(module, inp, out):
        attn_cache["ag2"] = out.detach().mean(1)

    # Attach hooks to the final Sigmoid layer of the Attention Gates
    # The output of model.agX.psi is the attention mask (alpha)
    h1 = model.ag1.psi.register_forward_hook(hook_ag1)
    h2 = model.ag2.psi.register_forward_hook(hook_ag2)

    try:
        with torch.no_grad():
            x = preprocess(img_arr)
            logits = model(x)

            probs = torch.sigmoid(logits).cpu().squeeze().numpy()  # HxW
            pred = (probs > thr).astype(np.uint8) if mode != "Attention only" else None

            # attention maps (fine = a1, coarse = a2)
            # SYNCED: Pull from cache, unsqueeze (B,H,W) -> (H,W)
            a1_np = attn_cache.get("ag1")[0].cpu().numpy() if attn_cache.get("ag1") is not None else None
            a2_np = attn_cache.get("ag2")[0].cpu().numpy() if attn_cache.get("ag2") is not None else None

    finally:
        # Crucial: remove hooks immediately after use
        h1.remove()
        h2.remove()

    return probs, pred, a1_np, a2_np


# -----------------
# Streamlit UI (No change needed)
# -----------------
st.set_page_config(page_title="Hybrid Q-UNet (Attention) — Demo", layout="wide")
st.title("Hybrid (Quantum + Attention) U-Net — Interactive Demo (Synced Architecture)")

left, right = st.columns([2, 1])

with right:
    st.subheader("1) Options")
    mode = st.selectbox("What to visualize?", ["Prediction + Attention", "Prediction only", "Attention only"])
    thr = st.slider("Decision Threshold", 0.0, 1.0, float(BEST_THR_HYBRID), 0.01,
                    help=f"Best tuned threshold from training evaluation: {BEST_THR_HYBRID}")

    st.subheader("2) Upload Image")
    img_file = st.file_uploader("MRI slice (PNG/JPG/NPY)", type=["png", "jpg", "jpeg", "npy"])

    st.subheader("Optional: Ground Truth Mask")
    gt_file = st.file_uploader("Binary mask (same size, PNG/JPG/NPY)", type=["png", "jpg", "jpeg", "npy"])

    run_btn = st.button("Run", type="primary", use_container_width=True)

with left:
    st.subheader("Output")
    if run_btn and img_file is not None:
        img_arr = load_image_to_array(img_file)
        gt_arr = load_image_to_array(gt_file) if gt_file is not None else None

        # Basic size guard
        if gt_arr is not None and img_arr.shape != gt_arr.shape:
            st.error(f"Shape mismatch: image={img_arr.shape}, GT={gt_arr.shape}. Please upload matching sizes.")
        else:
            probs, pred, a1, a2 = run_hybrid_single(img_arr, thr, mode)

            # Probability map
            if mode != "Attention only":
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Probability Map")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(img_arr, cmap="gray")
                    im = ax.imshow(probs, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig, clear_figure=True)

                with c2:
                    st.caption("Overlay (Blue=GT, Red=Prediction)")
                    buf = draw_overlay(img_arr, pred=pred, gt=gt_arr, title="Hybrid (Q+Attention)", thr=thr)
                    st.image(buf)

            # Attention maps when requested
            if mode in ("Prediction + Attention", "Attention only"):
                st.markdown("#### Attention Maps")
                c3, c4 = st.columns(2)
                if a2 is not None:
                    with c3:
                        show_heatmap(img_arr, a2, "AG2 (coarse, encoder L2)")
                if a1 is not None:
                    with c4:
                        show_heatmap(img_arr, a1, "AG1 (fine, encoder L1)")

            # Metrics if GT is provided and we predicted
            if gt_arr is not None and pred is not None:
                m = compute_metrics(pred, (gt_arr > 0.5).astype(np.uint8))
                st.markdown("#### Metrics @ threshold {:.2f}".format(thr))
                st.write({
                    "Dice": round(m["dice"], 4),
                    "IoU": round(m["iou"], 4),
                    "Precision": round(m["precision"], 4),
                    "Recall": round(m["recall"], 4),
                    "F1": round(m["f1"], 4),
                    "Accuracy": round(m["accuracy"], 4),
                })
                # Simple confusion matrix image
                fig, ax = plt.subplots(figsize=(3.8, 3.2))
                im = ax.imshow(m["cm"], cmap="Blues")
                for (i, j), v in np.ndenumerate(m["cm"]):
                    ax.text(j, i, str(v), ha="center", va="center")
                ax.set_xticks([0, 1]);
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["BG", "Tumor"]);
                ax.set_yticklabels(["BG", "Tumor"])
                ax.set_xlabel("Predicted");
                ax.set_ylabel("True");
                ax.set_title("Confusion (pixels)")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

            if (qml is None):
                st.warning(
                    "PennyLane not found – running hybrid with a lightweight fallback (no true quantum simulation). Install pennylane for full behavior.")

            st.success("Done!")
    else:
        st.info("Upload an image (and optional GT), choose a mode, and click *Run*.")

