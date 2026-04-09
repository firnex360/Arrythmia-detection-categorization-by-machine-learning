"""
ECG Arrhythmia Detection Web App
Flask backend — loads the trained ResNet1D model and exposes a /predict endpoint.

Model:   ecg_arrhythmia_model_v3.pt   (ResNet1D, 4-class: SR / AFIB / STACH / SBRAD)
History: ecg_arrhythmia_model_v3_history.json  (replaces old metadata.json)

Supported input formats
-----------------------
- .dat files: raw int16, interleaved 12 leads (N×12), slice/pad to TARGET_SAMPLES, scale ÷1000
- .mat files: 'val' key (or 'data'/'signal'/'ECG'/'ecg'), shape (12, N) or (N, 12).
  * Normalised to TARGET_SAMPLES (1000) by slicing or zero-padding.
- .pt  files: pre-processed PyTorch tensors of shape (12, 1000), as saved by the training pipeline.
- .png / .jpg / .jpeg / .bmp / .tiff: ECG strip images — the waveform is digitised from the image
  using a weighted column-centroid method.  Because a single flat image gives only one signal
  channel, that channel is replicated across all 12 leads.  Accuracy is lower than using raw data.

TARGET_SAMPLES = 1000  (PTB-XL 100 Hz × 10 s, same as training)
"""

import io
import json
import os
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from flask import Flask, request, jsonify, render_template

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "ecg_arrhythmia_model_v3.pt")
HISTORY_PATH  = os.path.join(BASE_DIR, "ecg_arrhythmia_model_v3_history.json")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Samples the model was trained on (PTB-XL 100 Hz × 10 s)
TARGET_SAMPLES = 1000

# ──────────────────────────────────────────────────────────────────────────────
# Load training history
# Keys: loss, accuracy, f1, val_loss, val_accuracy, val_f1
# ──────────────────────────────────────────────────────────────────────────────

with open(HISTORY_PATH, "r") as f:
    TRAINING_HISTORY = json.load(f)

# Class names as defined in learning_lastver.ipynb
# arrhythmia_map = {'SR': 0, 'AFIB': 1, 'STACH': 2, 'SBRAD': 3}
CLASS_NAMES = ["SR", "AFIB", "STACH", "SBRAD"]
N_LEADS     = 12
N_CLASSES   = len(CLASS_NAMES)

# Accepted file extensions
SIGNAL_EXTS = {".dat", ".mat", ".pt"}
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
ALL_EXTS    = SIGNAL_EXTS | IMAGE_EXTS

# ──────────────────────────────────────────────────────────────────────────────
# Rich label info
# ──────────────────────────────────────────────────────────────────────────────

_FULL_NAMES = {
    "SR":    "Sinus Rhythm",
    "AFIB":  "Atrial Fibrillation",
    "STACH": "Sinus Tachycardia",
    "SBRAD": "Sinus Bradycardia",
    "NORM":  "Normal Sinus Rhythm",
    "MI":    "Myocardial Infarction",
    "STTC":  "ST/T-Wave Change",
    "CD":    "Conduction Disturbance",
    "HYP":   "Hypertrophy",
    "TWC":   "T-Wave Change",
    "LBBB":  "Left Bundle Branch Block",
    "RBBB":  "Right Bundle Branch Block",
    "PAC":   "Premature Atrial Complex",
    "PVC":   "Premature Ventricular Complex",
}

_DESCRIPTIONS = {
    "SR":    "Normal sinus rhythm — the heart beats regularly at 60–100 bpm with impulses from the SA node. No significant arrhythmia detected.",
    "AFIB":  "Atrial fibrillation — chaotic electrical activity in the atria causes an irregular, often rapid rhythm. Increases stroke and heart failure risk.",
    "STACH": "Sinus tachycardia — heart rate >100 bpm with a regular rhythm originating from the SA node. Often triggered by exercise, fever, or stress.",
    "SBRAD": "Sinus bradycardia — heart rate <60 bpm with a regular sinus rhythm. Normal in athletes; may indicate conduction issues or medication effects in others.",
    "NORM":  "Normal ECG — no clinically significant abnormality detected.",
    "MI":    "Myocardial infarction — ST elevation or Q-wave changes indicating cardiac muscle damage from blocked coronary arteries.",
    "STTC":  "ST/T-wave change — ischemia, electrolyte imbalances, or repolarization abnormalities.",
    "CD":    "Conduction disturbance — delayed or blocked electrical propagation through the cardiac conduction system.",
    "HYP":   "Hypertrophy — thickening of the heart muscle, commonly from chronic hypertension or valvular disease.",
    "TWC":   "T-wave change — abnormal repolarization suggesting possible ischemia, electrolyte disturbance, or myocardial strain.",
    "LBBB":  "Left bundle branch block — altered QRS morphology from delayed left-sided conduction.",
    "RBBB":  "Right bundle branch block — widened QRS with right-sided conduction delay.",
    "PAC":   "Premature atrial complex — early atrial impulse generating a premature beat.",
    "PVC":   "Premature ventricular complex — wide QRS from an early ventricular impulse.",
}

_COLORS = {
    "SR":    "#10b981",
    "AFIB":  "#ef4444",
    "STACH": "#f59e0b",
    "SBRAD": "#3b82f6",
    "NORM":  "#10b981",
    "MI":    "#ef4444",
    "STTC":  "#a855f7",
    "CD":    "#6366f1",
    "HYP":   "#ec4899",
    "TWC":   "#a855f7",
    "LBBB":  "#14b8a6",
    "RBBB":  "#8b5cf6",
    "PAC":   "#06b6d4",
    "PVC":   "#84cc16",
}

# ──────────────────────────────────────────────────────────────────────────────
# Model definition — ResNet1D (identical to learning_lastver.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
                               stride=stride, padding=2, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, n_leads=12, n_classes=4):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1   = nn.Conv1d(n_leads, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1     = nn.BatchNorm1d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1  = self._make_layer(64,  blocks=2, stride=1)
        self.layer2  = self._make_layer(128, blocks=2, stride=2)
        self.layer3  = self._make_layer(256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.6)
        self.fc      = nn.Linear(256, n_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# Load model weights from ecg_arrhythmia_model_v3.pt
# ──────────────────────────────────────────────────────────────────────────────

state          = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
CKPT_N_CLASSES = state["fc.weight"].shape[0]
CKPT_N_LEADS   = state["conv1.weight"].shape[1]

if CKPT_N_CLASSES != N_CLASSES:
    print(f"[WARN] Checkpoint n_classes={CKPT_N_CLASSES} != expected {N_CLASSES}. "
          f"Adjusting CLASS_NAMES to first {CKPT_N_CLASSES} entries.")
    CLASS_NAMES = CLASS_NAMES[:CKPT_N_CLASSES]
    N_CLASSES   = CKPT_N_CLASSES

model = ResNet1D(n_leads=CKPT_N_LEADS, n_classes=CKPT_N_CLASSES).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"[OK] ecg_arrhythmia_model_v3.pt loaded — "
      f"{CKPT_N_LEADS} leads, {CKPT_N_CLASSES} classes: {CLASS_NAMES}  on {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# ECG preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_length(data_2d: np.ndarray) -> np.ndarray:
    """
    Ensure shape (n_leads, TARGET_SAMPLES).
    Matches notebook logic: slice if too long, zero-pad on the right if too short.
    """
    n_leads, n_samples = data_2d.shape
    if n_samples >= TARGET_SAMPLES:
        return data_2d[:, :TARGET_SAMPLES]
    pad = np.zeros((n_leads, TARGET_SAMPLES - n_samples), dtype=data_2d.dtype)
    return np.hstack([data_2d, pad])


def parse_dat(file_bytes: bytes):
    """Parse a PTB-XL-style .dat file (int16, interleaved 12 leads)."""
    raw    = np.frombuffer(file_bytes, dtype=np.int16)
    n_samp = len(raw) // 12
    data   = raw[: n_samp * 12].reshape(n_samp, 12)   # (N, 12)
    data   = data.T.astype(np.float32)                 # (12, N)
    data   = _normalize_length(data)                   # (12, 1000)
    x      = torch.tensor(data, dtype=torch.float32) / 1000.0
    return x, data


def parse_mat(file_bytes: bytes):
    """Parse a .mat ECG file — tries common key names used across datasets."""
    mat_data = loadmat(io.BytesIO(file_bytes))

    possible_keys = ["val", "data", "signal", "ECG", "ecg"]
    raw = None
    for key in possible_keys:
        if key in mat_data:
            raw = mat_data[key].astype(np.float32)
            break

    if raw is None:
        available = [k for k in mat_data.keys() if not k.startswith("_")]
        raise KeyError(f"No ECG data found in .mat file. Available keys: {available}")

    if raw.shape[0] != 12 and raw.shape[1] == 12:
        raw = raw.T
    if raw.shape[0] != 12:
        raise ValueError(f"Unexpected .mat shape {raw.shape}. Expected (12, N) or (N, 12).")

    if np.abs(raw).max() > 10:
        raw = raw / 1000.0
        raw_display = (raw * 1000.0).astype(np.float32)
    else:
        raw_display = (raw * 1000.0).astype(np.float32)

    data = _normalize_length(raw)
    x    = torch.tensor(data, dtype=torch.float32)
    return x, (raw_display if raw_display.shape == data.shape
               else (data * 1000.0).astype(np.float32))


def parse_pt(file_bytes: bytes):
    """
    Load a pre-processed PyTorch tensor (.pt) saved by the training pipeline.

    Expected shape: (12, 1000) — 12 leads × 1000 time-steps, already normalised.
    Also accepts (1, 1000) single-lead tensors (replicated to 12 leads) and
    bare 1-D tensors of length 1000.
    """
    buf = io.BytesIO(file_bytes)
    x   = torch.load(buf, map_location="cpu", weights_only=True)

    if not isinstance(x, torch.Tensor):
        raise ValueError(f".pt file must contain a torch.Tensor, got {type(x)}.")

    x = x.float()

    # Accept (12, 1000), (1, 1000), or (1000,)
    if x.ndim == 1:
        if x.shape[0] != TARGET_SAMPLES:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0).unsqueeze(0), size=TARGET_SAMPLES, mode="linear",
                align_corners=False
            ).squeeze(0).squeeze(0)
        x = x.unsqueeze(0).expand(CKPT_N_LEADS, -1).clone()  # (12, 1000)

    elif x.ndim == 2:
        n_leads_in, n_samp = x.shape
        # Normalise length
        if n_samp != TARGET_SAMPLES:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=TARGET_SAMPLES, mode="linear",
                align_corners=False
            ).squeeze(0)
        # Normalise lead count
        if n_leads_in == 1:
            x = x.expand(CKPT_N_LEADS, -1).clone()
        elif n_leads_in != CKPT_N_LEADS:
            raise ValueError(
                f".pt tensor has {n_leads_in} leads; model expects {CKPT_N_LEADS}."
            )
    else:
        raise ValueError(f"Unexpected tensor ndim={x.ndim}. Expected 1-D or 2-D tensor.")

    raw_display = (x.numpy() * 1000.0).astype(np.float32)
    return x, raw_display


def parse_image(file_bytes: bytes):
    """
    Digitise an ECG waveform from a raster image (PNG, JPG, BMP, TIFF, …).

    Algorithm
    ---------
    1. Convert to grayscale and invert so that dark waveform pixels become bright.
    2. Apply a small Gaussian blur column-wise to reduce grid/noise interference.
    3. For each pixel column, compute the brightness-weighted centroid row — this
       sub-pixel estimate is more robust than a plain argmax.
    4. Map row indices to a ±1 voltage-like signal (top of image = positive).
    5. Resample linearly to TARGET_SAMPLES (1000).
    6. Replicate the single extracted channel across all 12 model leads.

    Limitations
    -----------
    * Only one signal channel is extracted from the image, which is then replicated
      across all 12 leads.  The model was trained on real 12-lead recordings, so
      predictions from images are less reliable than from raw .dat / .mat / .pt files.
    * Works best with a plain single-strip ECG on a light background (printed or
      screen-captured).  Complex multi-panel printouts (12-lead layout) will yield
      a mixed signal that doesn't correspond cleanly to any individual lead.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("L")   # grayscale uint8
    img_arr = np.array(img, dtype=np.float32)               # (height, width)
    height, width = img_arr.shape

    # Invert: dark waveform → bright, light background → dark
    inverted = 255.0 - img_arr                              # (height, width)

    # Smooth each column slightly to suppress grid lines / JPEG artefacts
    smoothed = gaussian_filter1d(inverted, sigma=2.0, axis=0)  # along rows

    # Brightness-weighted centroid row per column
    row_indices  = np.arange(height, dtype=np.float32)[:, np.newaxis]  # (H, 1)
    weight_sum   = smoothed.sum(axis=0)                                 # (W,)
    weight_sum   = np.where(weight_sum > 0, weight_sum, 1.0)           # avoid /0
    centroid_row = (smoothed * row_indices).sum(axis=0) / weight_sum   # (W,)

    # Map rows to a ±1 signal: row 0 = top = +1, row H-1 = bottom = -1
    signal = (height / 2.0 - centroid_row) / (height / 2.0)           # (W,)

    # Resample to TARGET_SAMPLES via linear interpolation
    x_orig = np.linspace(0.0, 1.0, width)
    x_new  = np.linspace(0.0, 1.0, TARGET_SAMPLES)
    signal_resampled = np.interp(x_new, x_orig, signal).astype(np.float32)  # (1000,)

    # Replicate across CKPT_N_LEADS leads → (12, 1000)
    data        = np.tile(signal_resampled, (CKPT_N_LEADS, 1))         # (12, 1000)
    x_tensor    = torch.tensor(data, dtype=torch.float32)
    raw_display = (data * 1000.0).astype(np.float32)
    return x_tensor, raw_display


# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    uploaded = request.files["file"]
    filename  = uploaded.filename or ""
    ext       = os.path.splitext(filename)[1].lower()

    if ext not in ALL_EXTS:
        return jsonify({
            "error": (
                f"Unsupported file type '{ext}'. "
                f"Accepted formats: {', '.join(sorted(ALL_EXTS))}"
            )
        }), 400

    try:
        file_bytes = uploaded.read()

        image_warning = None

        if ext == ".dat":
            x_tensor, raw_signal = parse_dat(file_bytes)
        elif ext == ".mat":
            x_tensor, raw_signal = parse_mat(file_bytes)
        elif ext == ".pt":
            x_tensor, raw_signal = parse_pt(file_bytes)
        else:
            # Image path
            x_tensor, raw_signal = parse_image(file_bytes)
            image_warning = (
                "Prediction was made from an image. Only one signal channel could be "
                "extracted and was replicated across all 12 leads. Accuracy is lower "
                "than with raw ECG data (.dat / .mat / .pt). Use raw signal files for "
                "reliable clinical results."
            )

        # Add batch dimension → (1, 12, 1000)
        x_input = x_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits   = model(x_input)
            probs    = F.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())

        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx].item())

        class_probs = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(N_CLASSES)}

        LEAD_LABELS = ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
        all_leads = {LEAD_LABELS[i]: raw_signal[i, :].tolist()
                     for i in range(raw_signal.shape[0])}

        response = {
            "prediction":   pred_class,
            "full_name":    _FULL_NAMES.get(pred_class, pred_class),
            "description":  _DESCRIPTIONS.get(pred_class, ""),
            "color":        _COLORS.get(pred_class, "#38bdf8"),
            "confidence":   confidence,
            "class_probs":  class_probs,
            "class_colors": {c: _COLORS.get(c, "#38bdf8") for c in CLASS_NAMES},
            "class_names":  {c: _FULL_NAMES.get(c, c) for c in CLASS_NAMES},
            "all_leads":    all_leads,
            "filename":     filename,
        }

        if image_warning:
            response["warning"] = image_warning

        return jsonify(response)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
