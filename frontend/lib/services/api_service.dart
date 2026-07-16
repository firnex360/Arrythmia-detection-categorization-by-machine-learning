import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'package:frontend/core/config.dart';
import 'package:frontend/models/models.dart';
import 'package:frontend/core/session.dart';

/// Thin HTTP client for the Flask backend (`app.py`).
class ApiService {
  static Uri _u(String path) => Uri.parse('${AppConfig.baseUrl}$path');

  static Map<String, String> get _jsonHeaders => {
        'Content-Type': 'application/json',
        if (Session.isLoggedIn) 'Authorization': 'Bearer ${Session.token}',
      };

  static Map<String, String> get _authHeader =>
      {if (Session.isLoggedIn) 'Authorization': 'Bearer ${Session.token}'};

  static Map<String, dynamic> _decode(http.Response resp) {
    Map<String, dynamic> body;
    try {
      body = jsonDecode(resp.body) as Map<String, dynamic>;
    } catch (_) {
      throw ApiException('Unexpected response from server (${resp.statusCode}).');
    }
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw ApiException('${body['error'] ?? 'Request failed (${resp.statusCode}).'}');
    }
    return body;
  }

  // ── Health ──────────────────────────────────────────────────────────────
  static Future<Map<String, dynamic>> health() async {
    final resp = await http.get(_u('/health')).timeout(const Duration(seconds: 8));
    if (resp.statusCode != 200) {
      throw ApiException('Backend returned ${resp.statusCode}.');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  // ── Auth ────────────────────────────────────────────────────────────────
  static Future<Doctor> login(String username, String password) async {
    final resp = await http
        .post(_u('/auth/login'),
            headers: _jsonHeaders,
            body: jsonEncode({'username': username, 'password': password}))
        .timeout(const Duration(seconds: 15));
    final body = _decode(resp);
    final doctor = Doctor.fromJson(body['doctor'] as Map<String, dynamic>);
    Session.set('${body['token']}', doctor);
    return doctor;
  }

  static Future<Doctor> register(
      String username, String password, String name) async {
    final resp = await http
        .post(_u('/auth/register'),
            headers: _jsonHeaders,
            body: jsonEncode(
                {'username': username, 'password': password, 'name': name}))
        .timeout(const Duration(seconds: 15));
    final body = _decode(resp);
    final doctor = Doctor.fromJson(body['doctor'] as Map<String, dynamic>);
    Session.set('${body['token']}', doctor);
    return doctor;
  }

  static Future<void> logout() async {
    try {
      await http
          .post(_u('/auth/logout'), headers: _jsonHeaders)
          .timeout(const Duration(seconds: 8));
    } catch (_) {
      // Even if the network call fails, drop the local session.
    }
    Session.clear();
  }

  // ── Patients ────────────────────────────────────────────────────────────
  static Future<List<Patient>> listPatients() async {
    final resp = await http
        .get(_u('/patients'), headers: _authHeader)
        .timeout(const Duration(seconds: 15));
    final body = _decode(resp);
    return (body['patients'] as List)
        .map((e) => Patient.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  static Map<String, dynamic> _patientBody({
    String? cedula,
    String? firstName,
    String? lastName,
    String? dob,
    String? gender,
    String? notes,
  }) =>
      {
        'cedula': cedula,
        'first_name': firstName,
        'last_name': lastName,
        'dob': dob,
        'gender': gender,
        'notes': notes,
      };

  static Future<Patient> createPatient({
    String? cedula,
    String? firstName,
    String? lastName,
    String? dob,
    String? gender,
    String? notes,
  }) async {
    final resp = await http
        .post(_u('/patients'),
            headers: _jsonHeaders,
            body: jsonEncode(_patientBody(
                cedula: cedula,
                firstName: firstName,
                lastName: lastName,
                dob: dob,
                gender: gender,
                notes: notes)))
        .timeout(const Duration(seconds: 15));
    return Patient.fromJson(_decode(resp)['patient'] as Map<String, dynamic>);
  }

  static Future<Patient> updatePatient(
    int id, {
    String? cedula,
    String? firstName,
    String? lastName,
    String? dob,
    String? gender,
    String? notes,
  }) async {
    final resp = await http
        .put(_u('/patients/$id'),
            headers: _jsonHeaders,
            body: jsonEncode(_patientBody(
                cedula: cedula,
                firstName: firstName,
                lastName: lastName,
                dob: dob,
                gender: gender,
                notes: notes)))
        .timeout(const Duration(seconds: 15));
    return Patient.fromJson(_decode(resp)['patient'] as Map<String, dynamic>);
  }

  static Future<void> deletePatient(int id) async {
    final resp = await http
        .delete(_u('/patients/$id'), headers: _authHeader)
        .timeout(const Duration(seconds: 15));
    _decode(resp);
  }

  /// Returns (patient, records) for the detail screen.
  static Future<(Patient, List<EcgRecord>)> getPatient(int id) async {
    final resp = await http
        .get(_u('/patients/$id'), headers: _authHeader)
        .timeout(const Duration(seconds: 15));
    final body = _decode(resp);
    final patient = Patient.fromJson(body['patient'] as Map<String, dynamic>);
    final records = (body['records'] as List)
        .map((e) => EcgRecord.fromJson(e as Map<String, dynamic>))
        .toList();
    return (patient, records);
  }

  // ── Analysis + records ──────────────────────────────────────────────────
  /// Uploads an ECG file for [patientId], runs the model, and stores the result.
  /// Returns (record, alreadyExisted). If the same file was analysed before, the
  /// stored record is returned unchanged.
  static Future<(EcgRecord, bool)> analyzeForPatient({
    required int patientId,
    required Uint8List bytes,
    required String filename,
  }) async {
    final request = http.MultipartRequest('POST', _u('/patients/$patientId/analyze'))
      ..headers.addAll(_authHeader)
      ..files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));

    final streamed = await request.send().timeout(const Duration(seconds: 60));
    final resp = await http.Response.fromStream(streamed);
    final body = _decode(resp);
    return (
      EcgRecord.fromJson(body['record'] as Map<String, dynamic>),
      body['already_existed'] == true,
    );
  }

  static Future<EcgRecord> getRecord(int id) async {
    final resp = await http
        .get(_u('/records/$id'), headers: _authHeader)
        .timeout(const Duration(seconds: 15));
    return EcgRecord.fromJson(_decode(resp)['record'] as Map<String, dynamic>);
  }

  static Future<EcgRecord> updateRecordNotes(int id, String notes) async {
    final resp = await http
        .put(_u('/records/$id/notes'),
            headers: _jsonHeaders, body: jsonEncode({'doctor_notes': notes}))
        .timeout(const Duration(seconds: 15));
    return EcgRecord.fromJson(_decode(resp)['record'] as Map<String, dynamic>);
  }

  /// Records the doctor's verdict on a prediction. [verdict] is 'correct',
  /// 'incorrect' or null (clears it); [trueLabel] is the actual class when
  /// marking incorrect.
  static Future<EcgRecord> setRecordVerdict(
    int id, {
    required String? verdict,
    String? trueLabel,
  }) async {
    final resp = await http
        .put(_u('/records/$id/verdict'),
            headers: _jsonHeaders,
            body: jsonEncode({'verdict': verdict, 'true_label': trueLabel}))
        .timeout(const Duration(seconds: 15));
    return EcgRecord.fromJson(_decode(resp)['record'] as Map<String, dynamic>);
  }

  // ── Dashboard ───────────────────────────────────────────────────────────
  static Future<DashboardData> dashboard({
    String? from,
    String? to,
    String? gender,
  }) async {
    final q = <String, String>{};
    if (from != null && from.isNotEmpty) q['from'] = from;
    if (to != null && to.isNotEmpty) q['to'] = to;
    if (gender != null && gender.isNotEmpty) q['gender'] = gender;
    final uri = Uri.parse('${AppConfig.baseUrl}/dashboard')
        .replace(queryParameters: q.isEmpty ? null : q);
    final resp =
        await http.get(uri, headers: _authHeader).timeout(const Duration(seconds: 15));
    return DashboardData.fromJson(_decode(resp));
  }

  static Future<RiskOverview> risk() async {
    final resp = await http
        .get(_u('/risk'), headers: _authHeader)
        .timeout(const Duration(seconds: 15));
    return RiskOverview.fromJson(_decode(resp));
  }

  // ── Profile ─────────────────────────────────────────────────────────────
  static Future<Doctor> updateProfile({
    String? name,
    String? avatarColor,
    String? password,
  }) async {
    final resp = await http
        .put(_u('/me'),
            headers: _jsonHeaders,
            body: jsonEncode({
              if (name != null) 'name': name,
              if (avatarColor != null) 'avatar_color': avatarColor,
              if (password != null && password.isNotEmpty) 'password': password,
            }))
        .timeout(const Duration(seconds: 15));
    final doctor =
        Doctor.fromJson(_decode(resp)['doctor'] as Map<String, dynamic>);
    Session.doctor = doctor; // keep the session's copy fresh
    return doctor;
  }
}

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}
