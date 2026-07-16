import 'package:frontend/models/models.dart';

/// In-memory session for the logged-in doctor.
///
/// The *data* (patients, ECG history, notes) is persisted server-side in the
/// backend's SQLite database, so it survives restarts. The auth token itself is
/// kept only in memory — the doctor logs in again when the app is relaunched.
class Session {
  Session._();

  static String? token;
  static Doctor? doctor;

  static bool get isLoggedIn => token != null && token!.isNotEmpty;

  static void set(String newToken, Doctor newDoctor) {
    token = newToken;
    doctor = newDoctor;
  }

  static void clear() {
    token = null;
    doctor = null;
  }
}
