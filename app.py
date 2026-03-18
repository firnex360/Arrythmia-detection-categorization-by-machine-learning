"""
ECG Arrhythmia Detection Web App
Flask backend — loads the trained ECGModel and exposes a /predict endpoint.

Model: model.pt
Metadata: metadata.json (class_names, n_leads, n_classes, dropout_p, lead_index)

Key preprocessing notes
-----------------------
- .dat files: raw int16, interleaved 12 leads (N×12), slice/pad to TARGET_SAMPLES, scale ÷1000
- .mat files: 'val' key, shape (12, N).
  * If N == TARGET_SAMPLES  → use as-is
  * If N >  TARGET_SAMPLES  → downsample (scipy resample) to TARGET_SAMPLES so we
    keep the full 10-second window at the correct rate rather than slicing.
  * If N <  TARGET_SAMPLES  → zero-pad on the right
- TARGET_SAMPLES is inferred from what the training used (1000 for 100 Hz / 10 s).
"""

import io
import json
import os
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from flask import Flask, request, jsonify, render_template

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "model.pt")
METADATA_PATH  = os.path.join(BASE_DIR, "metadata.json")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Samples the model was trained on (PTB-XL 100 Hz × 10 s)
TARGET_SAMPLES = 1000

# ──────────────────────────────────────────────────────────────────────────────
# Load metadata
# ──────────────────────────────────────────────────────────────────────────────

with open(METADATA_PATH, "r") as f:
    meta = json.load(f)

META_CLASSES   = meta.get("class_names", [])   # e.g. ["SR","AFIB","STACH","SBRAD","TWC"]
META_N_LEADS   = meta.get("n_leads",   12)
META_N_CLASSES = meta.get("n_classes", len(META_CLASSES))
META_DROPOUT   = meta.get("dropout_p", 0.5)
META_LEAD_IDX  = meta.get("lead_index", None)

# ──────────────────────────────────────────────────────────────────────────────
# Rich label info for every class we might encounter
# ──────────────────────────────────────────────────────────────────────────────

_FULL_NAMES = {
    # 5-class set (learning_6_improved)
    "SR":    "Sinus Rhythm",
    "AFIB":  "Atrial Fibrillation",
    "STACH": "Sinus Tachycardia",
    "SBRAD": "Sinus Bradycardia",
    "TWC":   "T-Wave Change",
    # 12-class fallback (older PTB-XL superdiagnostic labels)
    "NORM":  "Normal Sinus Rhythm",
    "MI":    "Myocardial Infarction",
    "STTC":  "ST/T-Wave Change",
    "CD":    "Conduction Disturbance",
    "HYP":   "Hypertrophy",
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
    "TWC":   "T-wave change — abnormal repolarization suggesting possible ischemia, electrolyte disturbance, or myocardial strain.",
    "NORM":  "Normal ECG — no clinically significant abnormality detected.",
    "MI":    "Myocardial infarction — ST elevation or Q-wave changes indicating cardiac muscle damage from blocked coronary arteries.",
    "STTC":  "ST/T-wave change — ischemia, electrolyte imbalances, or repolarization abnormalities.",
    "CD":    "Conduction disturbance — delayed or blocked electrical propagation through the cardiac conduction system.",
    "HYP":   "Hypertrophy — thickening of the heart muscle, commonly from chronic hypertension or valvular disease.",
    "LBBB":  "Left bundle branch block — altered QRS morphology from delayed left-sided conduction.",
    "RBBB":  "Right bundle branch block — widened QRS with right-sided conduction delay.",
    "PAC":   "Premature atrial complex — early atrial impulse generating a premature beat.",
    "PVC":   "Premature ventricular complex — wide QRS from an early ventricular impulse.",
}

_COLORS = {
    "SR":    "#10b981",
    "NORM":  "#10b981",
    "AFIB":  "#ef4444",
    "MI":    "#ef4444",
    "STACH": "#f59e0b",
    "SBRAD": "#3b82f6",
    "TWC":   "#a855f7",
    "STTC":  "#a855f7",
    "CD":    "#6366f1",
    "HYP":   "#ec4899",
    "LBBB":  "#14b8a6",
    "RBBB":  "#8b5cf6",
    "PAC":   "#06b6d4",
    "PVC":   "#84cc16",
}

# ──────────────────────────────────────────────────────────────────────────────
# Model definition  (identical to notebook)
# ──────────────────────────────────────────────────────────────────────────────

class ECGModel(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_leads, 32,  kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,      64,  kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,      128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool        = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1     = nn.Linear(128 * 4, 256)
        self.fc2     = nn.Linear(256, 128)
        self.out     = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.out(x)


# ──────────────────────────────────────────────────────────────────────────────
# Load model — read actual n_classes from checkpoint; trust checkpoint over metadata
# ──────────────────────────────────────────────────────────────────────────────

state           = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
CKPT_N_CLASSES  = state["out.weight"].shape[0]
CKPT_N_LEADS    = state["conv1.weight"].shape[1]

# If n_classes matches metadata, use metadata class names.
# Otherwise auto-assign generic names (shouldn't happen with a correct model file).
if CKPT_N_CLASSES == META_N_CLASSES and META_CLASSES:
    CLASS_NAMES = META_CLASSES
else:
    # Fall back: generic names up to checkpoint size
    _fallback_12 = ["SR", "AFIB", "STACH", "SBRAD", "TWC",
                    "NORM", "MI", "STTC", "CD", "HYP", "LBBB", "RBBB"]
    CLASS_NAMES = _fallback_12[:CKPT_N_CLASSES]
    print(f"[WARN] Metadata n_classes={META_N_CLASSES} != checkpoint n_classes={CKPT_N_CLASSES}. "
          f"Using auto-assigned class names: {CLASS_NAMES}")

model = ECGModel(n_leads=CKPT_N_LEADS, n_classes=CKPT_N_CLASSES,
                 dropout_p=META_DROPOUT).to(DEVICE)
model.load_state_dict(state)
model.eval()
print(f"[OK] model.pt loaded — {CKPT_N_LEADS} leads, {CKPT_N_CLASSES} classes: {CLASS_NAMES}  on {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# ECG preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_length(data_2d: np.ndarray) -> np.ndarray:
    """
    Ensure shape (n_leads, TARGET_SAMPLES).

    Matches the notebook's exact logic:
    - If n_samples >= TARGET_SAMPLES → slice the FIRST TARGET_SAMPLES columns.
      This is correct because the training files (PTB-XL 100 Hz) are already 1000
      samples, and longer files (e.g. 500 Hz HR version) also start at sample 0.
    - If n_samples <  TARGET_SAMPLES → zero-pad on the right.
    NOTE: Do NOT use scipy.signal.resample here — it changes the signal shape
    relative to training data and produces wrong predictions.
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
    """Parse a .mat ECG file with a 'val' key."""
    mat_data = loadmat(io.BytesIO(file_bytes))
    if "val" not in mat_data:
        raise KeyError("No 'val' key found in .mat file.")

    data = mat_data["val"].astype(np.float32)

    # Ensure shape (12, N)
    if data.shape[0] != 12 and data.shape[1] == 12:
        data = data.T
    if data.shape[0] != 12:
        raise ValueError(f"Unexpected .mat shape {data.shape}. Expected (12, N).")

    data = _normalize_length(data)
    x    = torch.tensor(data, dtype=torch.float32) / 1000.0
    return x, data




# ──────────────────────────────────────────────────────────────────────────────
# Optional: single-lead selection (mirrors LEAD_INDEX training option)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_lead_selection(x: torch.Tensor) -> torch.Tensor:
    """If META_LEAD_IDX is set, extract that single lead axis → (1, T)."""
    if META_LEAD_IDX is not None:
        x = x[META_LEAD_IDX].unsqueeze(0)   # (1, T)
    return x


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

    if ext not in (".dat", ".mat"):
        return jsonify({"error": "Unsupported file type. Please upload a .dat or .mat file."}), 400

    try:
        file_bytes = uploaded.read()

        if ext == ".dat":
            x_tensor, raw_signal = parse_dat(file_bytes)
        else:
            x_tensor, raw_signal = parse_mat(file_bytes)

        # Optional single-lead selection
        x_tensor = _apply_lead_selection(x_tensor)

        # Add batch dimension → (1, leads, 1000)
        x_tensor = x_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits   = model(x_tensor)
            probs    = F.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())

        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx].item())

        # Only expose the meaningful classes (defined in metadata).
        # The checkpoint may have more output nodes than active classes (e.g. 12 nodes but only 5 trained labels).
        N_DISPLAY   = min(META_N_CLASSES, CKPT_N_CLASSES) if META_N_CLASSES else CKPT_N_CLASSES
        DISP_NAMES  = CLASS_NAMES[:N_DISPLAY]

        # Re-normalise probabilities over the displayed slice so they sum to 1
        disp_probs_raw = np.array([float(probs[i].item()) for i in range(N_DISPLAY)])
        total = disp_probs_raw.sum()
        if total > 0:
            disp_probs_raw = disp_probs_raw / total

        class_probs = {DISP_NAMES[i]: float(disp_probs_raw[i]) for i in range(N_DISPLAY)}

        # Re-derive prediction from the displayed slice
        best_disp_idx = int(disp_probs_raw.argmax())
        pred_class = DISP_NAMES[best_disp_idx]
        confidence = float(disp_probs_raw[best_disp_idx])

        # Send all 12 lead signals for multi-lead visualization in the UI.
        # Each lead is a list of 1000 floats (raw ADC units, not divided by 1000).
        LEAD_LABELS = ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
        all_leads = {LEAD_LABELS[i]: raw_signal[i, :].tolist()
                     for i in range(raw_signal.shape[0])}

        return jsonify({
            "prediction":   pred_class,
            "full_name":    _FULL_NAMES.get(pred_class, pred_class),
            "description":  _DESCRIPTIONS.get(pred_class, ""),
            "color":        _COLORS.get(pred_class, "#38bdf8"),
            "confidence":   confidence,
            "class_probs":  class_probs,
            "class_colors": {c: _COLORS.get(c, "#38bdf8") for c in DISP_NAMES},
            "class_names":  {c: _FULL_NAMES.get(c, c) for c in DISP_NAMES},
            "all_leads":    all_leads,
            "filename":     filename,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
