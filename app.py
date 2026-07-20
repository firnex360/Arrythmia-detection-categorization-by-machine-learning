"""
ECG Arrhythmia Detection Web App
Flask backend — loads the trained ResNet1D model and exposes a /predict endpoint.

Model:   ecg_arrhythmia_model_v5_mita.pt   (ResNet1D, 4-class: SR / AFIB / STACH / SBRAD)
History: ecg_arrhythmia_model_v5_history_mita.json

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
import sys
import tempfile
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps

from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from flask import Flask, request, jsonify, render_template, g

import db

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "ecg_arrhythmia_model_v5_mita.pt")
HISTORY_PATH  = os.path.join(BASE_DIR, "ecg_arrhythmia_model_v5_history_mita.json")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Samples the model was trained on (PTB-XL 100 Hz × 10 s)
TARGET_SAMPLES = 1000

# ──────────────────────────────────────────────────────────────────────────────
# ECGtizer (image / PDF → signal) configuration
#
# ECGtizer is loaded from a local repo folder dropped into this project as
# `./ecgtizer` (same layout as testing-ecgtizer-v2.ipynb).  If that folder isn't
# present we fall back to whatever `ecgtizer` is importable from the environment.
# The import is done lazily inside parse_image_ecgtizer(), so the server still
# starts — and .pt / .mat / .dat still work — even when ECGtizer isn't installed.
# ──────────────────────────────────────────────────────────────────────────────

ECGTIZER_REPO   = os.path.join(BASE_DIR, "ecgtizer")
if os.path.isdir(ECGTIZER_REPO) and ECGTIZER_REPO not in sys.path:
    sys.path.insert(0, ECGTIZER_REPO)

# Digitisation settings — mirror the notebook's doctor-image case.
ECGTIZER_DPI     = 200            # notebook: 500 > 300 > 150 for clean renders
ECGTIZER_METHOD  = "fragmented"   # 'full' | 'fragmented' | 'lazy'
# Assumed clinical layout of the uploaded ECG. Only the classic 3x4 layout is
# wired up (LEAD_TIME_3X4); switch to LEAD_TIME_6X2 here if your prints use 6x2.
ECGTIZER_LAYOUT  = "3x4"

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
PDF_EXTS    = {".pdf"}                       # ECGtizer digitises clinical PDFs too
DIGITISE_EXTS = IMAGE_EXTS | PDF_EXTS        # everything routed through ECGtizer
ALL_EXTS    = SIGNAL_EXTS | DIGITISE_EXTS

# ──────────────────────────────────────────────────────────────────────────────
# Rich label info
# ──────────────────────────────────────────────────────────────────────────────

_FULL_NAMES = {
    "SR":    "Ritmo Sinusal",
    "AFIB":  "Fibrilación Auricular",
    "STACH": "Taquicardia Sinusal",
    "SBRAD": "Bradicardia Sinusal",
    "NORM":  "Ritmo Sinusal Normal",
    "MI":    "Infarto de Miocardio",
    "STTC":  "Cambio de Onda ST/T",
    "CD":    "Trastorno de Conducción",
    "HYP":   "Hipertrofia",
    "TWC":   "Cambio de Onda T",
    "LBBB":  "Bloqueo de Rama Izquierda",
    "RBBB":  "Bloqueo de Rama Derecha",
    "PAC":   "Complejo Auricular Prematuro",
    "PVC":   "Complejo Ventricular Prematuro",
}

_DESCRIPTIONS = {
    "SR":    "Ritmo sinusal normal — el corazón late de forma regular a 60–100 lpm con impulsos del nodo SA. No se detecta arritmia significativa.",
    "AFIB":  "Fibrilación auricular — la actividad eléctrica caótica en las aurículas produce un ritmo irregular, con frecuencia rápido. Aumenta el riesgo de ictus e insuficiencia cardíaca.",
    "STACH": "Taquicardia sinusal — frecuencia cardíaca >100 lpm con ritmo regular originado en el nodo SA. Suele deberse a ejercicio, fiebre o estrés.",
    "SBRAD": "Bradicardia sinusal — frecuencia cardíaca <60 lpm con ritmo sinusal regular. Normal en atletas; en otros puede indicar problemas de conducción o efecto de medicamentos.",
    "NORM":  "ECG normal — sin anomalías clínicamente significativas.",
    "MI":    "Infarto de miocardio — elevación del ST o cambios en la onda Q que indican daño del músculo cardíaco por arterias coronarias obstruidas.",
    "STTC":  "Cambio de onda ST/T — isquemia, desequilibrios electrolíticos o anomalías de repolarización.",
    "CD":    "Trastorno de conducción — propagación eléctrica retrasada o bloqueada en el sistema de conducción cardíaco.",
    "HYP":   "Hipertrofia — engrosamiento del músculo cardíaco, comúnmente por hipertensión crónica o enfermedad valvular.",
    "TWC":   "Cambio de onda T — repolarización anómala que sugiere posible isquemia, alteración electrolítica o sobrecarga miocárdica.",
    "LBBB":  "Bloqueo de rama izquierda — morfología del QRS alterada por conducción izquierda retrasada.",
    "RBBB":  "Bloqueo de rama derecha — QRS ensanchado con retraso de conducción derecho.",
    "PAC":   "Complejo auricular prematuro — impulso auricular temprano que genera un latido prematuro.",
    "PVC":   "Complejo ventricular prematuro — QRS ancho por un impulso ventricular temprano.",
}

# Características clínicas / de la señal propias de cada clase.  Este es el
# diccionario del "por qué" para la interfaz: explica, en lenguaje sencillo, en
# qué rasgos del ECG tiende a apoyarse el modelo al decidir un veredicto.  (Son
# los rasgos morfológicos conocidos de cada ritmo — ayudan a interpretar el mapa
# de calor Grad-CAM, que resalta *dónde* del trazado se concentró el modelo.)
_KEY_FEATURES = {
    "SR": [
        "Intervalos R-R regulares (espaciado uniforme entre latidos)",
        "Onda P clara y positiva antes de cada complejo QRS",
        "Frecuencia entre 60 y 100 lpm",
        "Morfología del QRS estrecha y constante",
    ],
    "AFIB": [
        "Intervalos R-R irregularmente irregulares (sin patrón repetitivo)",
        "Ausencia de ondas P — reemplazadas por ondas de fibrilación (f) caóticas",
        "Línea de base ondulante entre los complejos QRS",
        "Con frecuencia, respuesta ventricular rápida",
    ],
    "STACH": [
        "Frecuencia rápida (>100 lpm) con latidos muy juntos",
        "Intervalos R-R regulares pese a la velocidad",
        "Onda P normal precediendo cada QRS",
        "Segmento T-P acortado (menos reposo entre latidos)",
    ],
    "SBRAD": [
        "Frecuencia lenta (<60 lpm) con latidos muy separados",
        "Intervalos R-R regulares",
        "Onda P normal y positiva antes de cada QRS",
        "Segmento T-P largo y plano (reposo prolongado entre latidos)",
    ],
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
print(f"[OK] {os.path.basename(MODEL_PATH)} loaded — "
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


# ══════════════════════════════════════════════════════════════════════════════
#  ▞▞  IMAGE / PDF  →  .pt  CONVERSION  via  ECGtizer  ▞▞
#
#  This is the real digitiser, ported from testing-ecgtizer-v2.ipynb.  It takes a
#  photo/scan/PDF of a standard 12-lead clinical ECG (3x4 "classic" layout),
#  reconstructs all 12 leads, and returns a (12, 1000) tensor in millivolts —
#  exactly the shape/scale the model was trained on, so inference, Grad-CAM and
#  the Flutter UI all work unchanged.
#
#  Requirements (see README):
#    * Drop the ECGtizer repo into ./ecgtizer  (or pip-install it).
#    * Its deps must be available: opencv-python (cv2), PyMuPDF/fitz, scipy, etc.
#    * The image must be a proper clinical printout, 3x4 layout, high resolution
#      (>= ~1500x800, ideally 2000+ wide) or fewer than 12 leads will be found.
#
#  The crude single-lead `parse_image` above is kept only as an emergency fallback
#  (see IMAGE_FALLBACK_CENTROID) for when ECGtizer can't be loaded.
# ══════════════════════════════════════════════════════════════════════════════

# If ECGtizer is unavailable, fall back to the rough centroid digitiser instead
# of failing.  Off by default because that fallback is not clinically meaningful.
IMAGE_FALLBACK_CENTROID = False





def parse_image_ecgtizer(file_bytes: bytes, ext: str):
    """
    Digitise a clinical ECG image/PDF into a (12, 1000) tensor using ECGtizer.

    Returns (x_tensor, raw_display) just like the other parse_* helpers.
    Raises RuntimeError with an actionable message if ECGtizer isn't available
    or the image can't be digitised into 12 leads.
    """
    try:
        from ecgtizer import ECGtizer
        
        # Add ecgtizer-work to path so we can import our standalone script logic
        ECGTIZER_WORK = os.path.join(BASE_DIR, "ecgtizer-work")
        if ECGTIZER_WORK not in sys.path:
            sys.path.insert(0, ECGTIZER_WORK)
            
        from ecgtizer_v3 import ecgtizer_to_ptbxl
    except Exception as exc:                                   # not installed
        raise RuntimeError(
            "ECGtizer is not available, so ECG images/PDFs can't be digitised. "
            "Drop the ECGtizer repo into ./ecgtizer (and install its dependencies: "
            "opencv-python, PyMuPDF, scipy), or upload a raw .pt / .mat / .dat file "
            f"instead. (import error: {exc})"
        )

    # ECGtizer reads from a path, so persist the upload to a temp file first.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        ecg = ECGtizer(
            file=tmp_path,
            dpi=ECGTIZER_DPI,
            extraction_method=ECGTIZER_METHOD,
            verbose=False,
        )
        leads = ecg.extracted_lead

        if not isinstance(leads, dict) or len(leads) < 12:
            n = len(leads) if isinstance(leads, dict) else 0
            raise RuntimeError(
                f"ECGtizer only extracted {n}/12 leads. This usually means the "
                "image resolution is too low or the layout isn't the expected "
                f"{ECGTIZER_LAYOUT} clinical format. Use a sharper scan "
                "(>= ~2000 px wide) of a standard 12-lead printout."
            )

        signal = ecgtizer_to_ptbxl(leads, target_samples=TARGET_SAMPLES, layout="auto")  # (1000, 12) mV
        data   = signal.T.astype(np.float32)                          # (12, 1000)

        x_tensor    = torch.tensor(data, dtype=torch.float32)
        raw_display = (data * 1000.0).astype(np.float32)
        return x_tensor, raw_display

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM  — explains *where* in the trace the model focused
# ──────────────────────────────────────────────────────────────────────────────

def compute_gradcam(x_input: torch.Tensor, class_idx: int) -> list:
    """
    Grad-CAM for the 1-D ResNet.

    Hooks the last residual stage (`model.layer3`), captures its activations and
    the gradient of the target class score w.r.t. those activations, then builds
    a per-time-step importance curve resampled to TARGET_SAMPLES and normalised
    to 0..1.  A value near 1 means "the model leaned heavily on this part of the
    signal when choosing `class_idx`".

    Returns a plain Python list of length TARGET_SAMPLES.
    """
    activations = {}
    gradients   = {}

    def fwd_hook(_module, _inp, out):
        activations["value"] = out.detach()

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = model.layer3.register_forward_hook(fwd_hook)
    # full_backward_hook is the modern API; fall back for older torch builds
    try:
        h2 = model.layer3.register_full_backward_hook(bwd_hook)
    except AttributeError:                                    # pragma: no cover
        h2 = model.layer3.register_backward_hook(bwd_hook)

    try:
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)

        x = x_input.clone().requires_grad_(True)
        logits = model(x)                                    # (1, n_classes)
        score  = logits[0, class_idx]
        score.backward()

        acts  = activations["value"][0]                      # (C, L)
        grads = gradients["value"][0]                        # (C, L)

        # Channel weights = global-average-pooled gradients
        weights = grads.mean(dim=1, keepdim=True)            # (C, 1)
        cam = F.relu((weights * acts).sum(dim=0))            # (L,)

        # Resample the coarse CAM up to the full signal length
        cam = cam.view(1, 1, -1)
        cam = F.interpolate(cam, size=TARGET_SAMPLES, mode="linear",
                            align_corners=False).view(-1)

        # Normalise 0..1
        cam = cam - cam.min()
        maxv = cam.max()
        if maxv > 0:
            cam = cam / maxv

        if was_training:
            model.train()
        return cam.cpu().tolist()

    finally:
        h1.remove()
        h2.remove()


# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB

# Create tables + seed the demo doctor (admin / admin) on startup.
db.init_db()
print("[OK] database ready at", db.DB_PATH)


@app.after_request
def _add_cors_headers(resp):
    """Allow the Flutter app (any origin/device) to call this API."""
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return resp


@app.before_request
def _handle_cors_preflight():
    """Answer browser CORS preflight (OPTIONS) so the web build can call the API."""
    if request.method == "OPTIONS":
        return ("", 204)


# ──────────────────────────────────────────────────────────────────────────────
# Authentication — opaque bearer tokens in the Authorization header
# ──────────────────────────────────────────────────────────────────────────────

def _bearer_token():
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def require_auth(fn):
    """Route decorator: 401s unless a valid session token is present. Sets g.doctor."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        doctor = db.get_doctor_by_token(_bearer_token())
        if doctor is None:
            return jsonify({"error": "Not authenticated. Please log in."}), 401
        g.doctor = doctor
        return fn(*args, **kwargs)
    return wrapper


@app.route("/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or {}
    try:
        doc = db.create_doctor(
            data.get("username", ""), data.get("password", ""), data.get("name", ""),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    token = db.create_session(doc["id"])
    return jsonify({"token": token, "doctor": db.doctor_public(doc)})


@app.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    doc = db.verify_login(data.get("username", ""), data.get("password", ""))
    if doc is None:
        return jsonify({"error": "Invalid username or password."}), 401
    token = db.create_session(doc["id"])
    return jsonify({"token": token, "doctor": db.doctor_public(doc)})


@app.route("/auth/logout", methods=["POST"])
@require_auth
def auth_logout():
    db.delete_session(_bearer_token())
    return jsonify({"ok": True})


@app.route("/me")
@require_auth
def me():
    return jsonify({"doctor": db.doctor_public(g.doctor)})


@app.route("/me", methods=["PUT"])
@require_auth
def update_me():
    """Doctor updates their own profile (name, avatar colour, optional password)."""
    d = request.get_json(silent=True) or {}
    doc = db.update_doctor(
        g.doctor["id"],
        name=d.get("name"),
        avatar_color=d.get("avatar_color"),
        password=d.get("password") or None,
    )
    if doc is None:
        return jsonify({"error": "Doctor not found."}), 404
    return jsonify({"doctor": doc})


# ──────────────────────────────────────────────────────────────────────────────
# Patients
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/patients", methods=["GET"])
@require_auth
def patients_list():
    return jsonify({"patients": db.list_patients(g.doctor["id"])})


@app.route("/patients", methods=["POST"])
@require_auth
def patients_create():
    d = request.get_json(silent=True) or {}
    try:
        patient = db.create_patient(g.doctor["id"], d)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"patient": patient})


@app.route("/patients/<int:patient_id>", methods=["GET"])
@require_auth
def patient_detail(patient_id):
    row = db.get_patient(patient_id, g.doctor["id"])
    if row is None:
        return jsonify({"error": "Patient not found."}), 404
    return jsonify({
        "patient": db.patient_public(row),
        "records": db.list_records(patient_id),
    })


@app.route("/patients/<int:patient_id>", methods=["PUT"])
@require_auth
def patient_update(patient_id):
    d = request.get_json(silent=True) or {}
    try:
        patient = db.update_patient(patient_id, g.doctor["id"], d)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if patient is None:
        return jsonify({"error": "Patient not found."}), 404
    return jsonify({"patient": patient})


@app.route("/patients/<int:patient_id>", methods=["DELETE"])
@require_auth
def patient_delete(patient_id):
    if not db.delete_patient(patient_id, g.doctor["id"]):
        return jsonify({"error": "Patient not found."}), 404
    return jsonify({"ok": True})


@app.route("/patients/<int:patient_id>/analyze", methods=["POST"])
@require_auth
def patient_analyze(patient_id):
    """Run the model on an uploaded file and store the result for this patient.

    De-duplicates by file hash: re-uploading the same file returns the stored
    record (already_existed=True) instead of re-running the model.
    """
    if db.get_patient(patient_id, g.doctor["id"]) is None:
        return jsonify({"error": "Patient not found."}), 404
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    uploaded = request.files["file"]
    filename = uploaded.filename or ""
    ext      = os.path.splitext(filename)[1].lower()
    if ext not in ALL_EXTS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. "
                     f"Accepted formats: {', '.join(sorted(ALL_EXTS))}"
        }), 400

    try:
        file_bytes = uploaded.read()
        fhash = db.file_hash(file_bytes)

        existing = db.find_record_by_hash(patient_id, fhash)
        if existing is not None:
            rec = db.record_public(existing, include_full=True)
            return jsonify({"record": rec, "already_existed": True})

        result = run_prediction(file_bytes, ext, filename)
        rec = db.create_record(patient_id, g.doctor["id"], filename, fhash, result)
        return jsonify({"record": rec, "already_existed": False})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Records — full result re-render + doctor notes/recommendations
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/records/<int:record_id>", methods=["GET"])
@require_auth
def record_get(record_id):
    rec = db.get_record(record_id, g.doctor["id"])
    if rec is None:
        return jsonify({"error": "Record not found."}), 404
    return jsonify({"record": rec})


@app.route("/records/<int:record_id>/notes", methods=["PUT"])
@require_auth
def record_notes(record_id):
    d = request.get_json(silent=True) or {}
    rec = db.update_record_notes(record_id, g.doctor["id"], d.get("doctor_notes", ""))
    if rec is None:
        return jsonify({"error": "Record not found."}), 404
    return jsonify({"record": rec})


@app.route("/records/<int:record_id>/verdict", methods=["PUT"])
@require_auth
def record_verdict(record_id):
    """Doctor confirms whether the model's prediction was correct.

    Body: {"verdict": "correct"|"incorrect"|null, "true_label": "AFIB" (optional)}
    """
    d = request.get_json(silent=True) or {}
    try:
        rec = db.set_record_verdict(
            record_id, g.doctor["id"], d.get("verdict"), d.get("true_label"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if rec is None:
        return jsonify({"error": "Record not found."}), 404
    return jsonify({"record": rec})


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard — global aggregates enriched with class colours/names
# ──────────────────────────────────────────────────────────────────────────────

def _class_meta():
    return {
        "class_names":  {c: _FULL_NAMES.get(c, c) for c in CLASS_NAMES},
        "class_colors": {c: _COLORS.get(c, "#38bdf8") for c in CLASS_NAMES},
        "class_order":  list(CLASS_NAMES),
    }


@app.route("/dashboard", methods=["GET"])
@require_auth
def dashboard():
    stats = db.dashboard_stats(
        from_date=request.args.get("from") or None,
        to_date=request.args.get("to") or None,
        gender=request.args.get("gender") or None,
    )
    stats.update(_class_meta())
    return jsonify(stats)


@app.route("/risk", methods=["GET"])
@require_auth
def risk():
    """Risk & alerts triage for the logged-in doctor's patients."""
    data = db.risk_overview(g.doctor["id"])
    data.update(_class_meta())
    return jsonify(data)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Lightweight probe so the Flutter app can confirm the backend is reachable."""
    return jsonify({
        "status":      "ok",
        "model":       os.path.basename(MODEL_PATH),
        "n_leads":     CKPT_N_LEADS,
        "class_names": CLASS_NAMES,
        "device":      str(DEVICE),
    })


def run_prediction(file_bytes: bytes, ext: str, filename: str) -> dict:
    """
    Core inference used by both /predict and the patient-linked analyse route.

    Parses the file into a (12, 1000) tensor, runs the model, computes Grad-CAM,
    and returns the full response dict.  Raises on unsupported types or parse
    errors so callers can translate them into HTTP responses.
    """
    image_warning = None

    if ext == ".dat":
        x_tensor, raw_signal = parse_dat(file_bytes)
    elif ext == ".mat":
        x_tensor, raw_signal = parse_mat(file_bytes)
    elif ext == ".pt":
        x_tensor, raw_signal = parse_pt(file_bytes)
    elif ext in DIGITISE_EXTS:
        # Image / PDF path — digitise all 12 leads with ECGtizer.
        try:
            x_tensor, raw_signal = parse_image_ecgtizer(file_bytes, ext)
            image_warning = (
                "Este ECG se digitalizó desde una imagen/PDF con ECGtizer, asumiendo un "
                f"formato clínico estándar {ECGTIZER_LAYOUT}. La calidad de extracción "
                "depende de la resolución y calidad de impresión — verifica los trazados "
                "contra el original antes de confiar en el resultado."
            )
        except RuntimeError:
            # ECGtizer unavailable or couldn't digitise. Only fall back to the
            # crude single-lead method if explicitly enabled; otherwise surface
            # the actionable error to the user.
            if IMAGE_FALLBACK_CENTROID:
                x_tensor, raw_signal = parse_image(file_bytes)
                image_warning = (
                    "ECGtizer no pudo digitalizar esta imagen, así que se usó una "
                    "aproximación de una sola derivación (un canal copiado a las 12). Esto "
                    "NO es confiable — sube un archivo .pt / .mat / .dat para un resultado real."
                )
            else:
                raise
    else:
        raise ValueError(f"Unsupported file type '{ext}'.")

    # Add batch dimension → (1, 12, 1000)
    x_input = x_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits   = model(x_input)
        probs    = F.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx].item())

    class_probs = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(N_CLASSES)}

    # Grad-CAM importance curve for the winning class (0..1, length 1000)
    gradcam = compute_gradcam(x_input, pred_idx)

    LEAD_LABELS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]
    all_leads = {LEAD_LABELS[i]: raw_signal[i, :].tolist()
                 for i in range(raw_signal.shape[0])}

    response = {
        "prediction":    pred_class,
        "full_name":     _FULL_NAMES.get(pred_class, pred_class),
        "description":   _DESCRIPTIONS.get(pred_class, ""),
        "key_features":  _KEY_FEATURES.get(pred_class, []),
        "color":         _COLORS.get(pred_class, "#38bdf8"),
        "confidence":    confidence,
        "class_probs":   class_probs,
        "class_colors":  {c: _COLORS.get(c, "#38bdf8") for c in CLASS_NAMES},
        "class_names":   {c: _FULL_NAMES.get(c, c) for c in CLASS_NAMES},
        "all_leads":     all_leads,
        "gradcam":       gradcam,
        "gradcam_lead":  "II",
        "filename":      filename,
    }

    if image_warning:
        response["warning"] = image_warning

    return response


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
        response = run_prediction(uploaded.read(), ext, filename)
        return jsonify(response)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # En producción (Render) el puerto lo asigna la plataforma en $PORT y el
    # servidor lo levanta gunicorn, no este bloque. En local se usa el 5000.
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
