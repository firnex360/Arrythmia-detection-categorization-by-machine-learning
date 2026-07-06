# ECG Arrhythmia Detector — Flutter frontend

A Flutter client for the ECG arrhythmia model. The machine-learning code stays
in **raw Python**: the trained ResNet1D (`ecg_arrhythmia_model_v5_mita.pt`) runs
inside the Flask backend (`../app.py`), and this app uploads a file and renders
the result. Nothing about the model was re-implemented in Dart.

## What it does

- **Import a recording** — `.pt`, `.mat`, or `.dat` 12-lead ECG files.
- **Take / pick an image** — capture an ECG printout with the camera or choose
  one from the gallery (mobile only). Images are digitised by the backend's
  placeholder pipeline; see the note below.
- **Verdict** — predicted arrhythmia (SR / AFIB / STACH / SBRAD) with confidence.
- **All-class probabilities** — bars showing how strongly every other class was
  considered.
- **Grad-CAM** — the ECG trace is coloured from cool (ignored) to hot (heavily
  weighted) so you can see *where* the model focused when it decided.
- **Why this result** — a plain-language description plus the characteristic
  features of that rhythm.

## Running it

1. **Start the backend** (from the project root, one level up):

   ```bash
   pip install -r requirements.txt
   python app.py            # serves http://0.0.0.0:5000
   ```

2. **Run the app**:

   ```bash
   cd frontend
   flutter pub get
   flutter run              # pick a device (Windows / Chrome / Android / iOS)
   ```

3. **Point the app at the backend.** Defaults:
   - Android emulator → `http://10.0.2.2:5000` (auto)
   - Desktop / web / iOS simulator → `http://127.0.0.1:5000` (auto)
   - **Physical phone** → tap the server icon (top-right) and enter your
     computer's LAN IP, e.g. `http://192.168.1.19:5000`.

   Use **Test backend connection** on the home screen to confirm it's reachable.

## Image / PDF → 12-lead signal (ECGtizer)

Images and PDFs of a standard 12-lead clinical ECG (3×4 "classic" layout) are
digitised into a real `(12, 1000)` signal by **ECGtizer**, ported from
`testing-ecgtizer-v2.ipynb` into `../app.py` (`parse_image_ecgtizer`). The
reconstructed signal runs through the same inference + Grad-CAM path as raw data.

To enable it on the backend:

1. Drop the ECGtizer repo into the project root as `./ecgtizer` (same as the
   notebook's `ECGTIZER_REPO`), or `pip install` it if you have a package build.
2. Install its dependencies: `pip install opencv-python PyMuPDF wfdb scipy`.

Notes:
- Best results need a **sharp, high-resolution** image (≈2000 px wide). Low-res
  scans may yield fewer than 12 leads, in which case the backend returns a clear
  error asking for a better image.
- Only the **3×4 layout** is wired up. For 6×2 prints, set `ECGTIZER_LAYOUT = "6x2"`
  in `../app.py`.
- If ECGtizer isn't installed, `.pt / .mat / .dat` still work; image uploads
  return an actionable "ECGtizer not available" message. A crude single-lead
  fallback exists behind `IMAGE_FALLBACK_CENTROID` in `app.py` (off by default).

## Where things live

| File | Purpose |
|------|---------|
| `lib/config.dart` | Backend URL + accepted file types |
| `lib/api_service.dart` | HTTP client for `/health` and `/predict` |
| `lib/models.dart` | Parses the backend JSON response |
| `lib/home_screen.dart` | Import / camera / server settings |
| `lib/result_screen.dart` | Verdict, probabilities, explanation |
| `lib/widgets/gradcam_ecg.dart` | Grad-CAM coloured ECG painter |
| `lib/widgets/probability_bars.dart` | Per-class probability bars |
