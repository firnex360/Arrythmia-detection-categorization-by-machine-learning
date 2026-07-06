import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'config.dart';
import 'models.dart';

/// Thin HTTP client for the Flask backend (`app.py`).
class ApiService {
  /// Confirms the backend is up and the model is loaded.
  static Future<Map<String, dynamic>> health() async {
    final uri = Uri.parse('${AppConfig.baseUrl}/health');
    final resp = await http.get(uri).timeout(const Duration(seconds: 8));
    if (resp.statusCode != 200) {
      throw ApiException('Backend returned ${resp.statusCode}.');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  /// Uploads [bytes] (an ECG file or image) named [filename] to `/predict`
  /// and returns the parsed [PredictionResult].
  static Future<PredictionResult> predict({
    required Uint8List bytes,
    required String filename,
  }) async {
    final uri = Uri.parse('${AppConfig.baseUrl}/predict');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));

    final streamed =
        await request.send().timeout(const Duration(seconds: 60));
    final resp = await http.Response.fromStream(streamed);

    Map<String, dynamic> body;
    try {
      body = jsonDecode(resp.body) as Map<String, dynamic>;
    } catch (_) {
      throw ApiException('Unexpected response from server (${resp.statusCode}).');
    }

    if (resp.statusCode != 200) {
      throw ApiException('${body['error'] ?? 'Prediction failed.'}');
    }
    return PredictionResult.fromJson(body);
  }
}

class ApiException implements Exception {
  final String message;
  ApiException(this.message);
  @override
  String toString() => message;
}
