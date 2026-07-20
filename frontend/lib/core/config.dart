import 'package:flutter/foundation.dart';

/// App-wide configuration.
///
/// The heavy lifting (loading `ecg_arrhythmia_model_v5_mita.pt`, running the
/// ResNet1D, computing Grad-CAM) happens in the Python/Flask backend `app.py`.
/// Flutter is a thin client that uploads a file and renders the JSON response,
/// so the machine-learning code stays in raw Python — no Dart re-implementation.
class AppConfig {
  AppConfig._();

  /// URL del backend Flask, fijada al compilar.
  ///
  /// En producción se pasa la URL pública de Render:
  ///     flutter build web --dart-define=API_URL=https://siemia-api.onrender.com
  ///
  /// Si no se pasa, queda vacía y se usan los valores locales de desarrollo.
  static const String _buildTimeUrl = String.fromEnvironment('API_URL');

  /// Base URL of the Flask backend.
  ///
  /// Defaults are chosen so the app "just works" in the common cases:
  ///   * API_URL definida al compilar -> esa URL (despliegue en Render).
  ///   * Android emulator  -> 10.0.2.2 maps to the host machine's localhost.
  ///   * Everything else   -> localhost.
  ///
  /// On a real phone, set this to your computer's LAN IP (e.g.
  /// http://192.168.1.20:5000) from the in-app server settings dialog.
  static String baseUrl = _defaultBaseUrl();

  static String _defaultBaseUrl() {
    // La URL de compilación gana: es la que se usa en el despliegue.
    if (_buildTimeUrl.isNotEmpty) return _buildTimeUrl;

    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:5000';
    }
    return 'http://127.0.0.1:5000';
  }

  /// File extensions the doctor can upload — raw ECG signal data only.
  /// (Image/photo input was removed from the app.)
  static const List<String> signalExtensions = ['pt', 'mat', 'dat'];

  static List<String> get allExtensions => signalExtensions;
}
