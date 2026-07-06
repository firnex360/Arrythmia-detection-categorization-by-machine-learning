import 'package:flutter/material.dart';

/// Parsed result of a `/predict` call to the Flask backend.
class PredictionResult {
  final String prediction; // short code, e.g. "AFIB"
  final String fullName; // e.g. "Atrial Fibrillation"
  final String description; // plain-language summary of the rhythm
  final List<String> keyFeatures; // "what the model looked at" for this class
  final Color color; // accent colour for the winning class
  final double confidence; // 0..1 for the winning class

  /// Probability the model assigned to every class (code -> 0..1).
  final Map<String, double> classProbs;
  final Map<String, Color> classColors; // code -> colour
  final Map<String, String> classNames; // code -> full name

  /// Raw ECG traces per lead label (e.g. "II" -> [...samples]).
  final Map<String, List<double>> allLeads;

  /// Grad-CAM importance per time-step (0..1), aligned to [gradcamLead].
  final List<double> gradcam;
  final String gradcamLead; // which lead the heat-map is drawn over
  final String filename;
  final String? warning; // set when input was an image (low reliability)

  PredictionResult({
    required this.prediction,
    required this.fullName,
    required this.description,
    required this.keyFeatures,
    required this.color,
    required this.confidence,
    required this.classProbs,
    required this.classColors,
    required this.classNames,
    required this.allLeads,
    required this.gradcam,
    required this.gradcamLead,
    required this.filename,
    this.warning,
  });

  /// Classes sorted by probability, highest first.
  List<MapEntry<String, double>> get sortedProbs {
    final entries = classProbs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return entries;
  }

  static Color _parseColor(String? hex, [Color fallback = const Color(0xFF38BDF8)]) {
    if (hex == null) return fallback;
    var h = hex.replaceFirst('#', '');
    if (h.length == 6) h = 'FF$h';
    final value = int.tryParse(h, radix: 16);
    return value == null ? fallback : Color(value);
  }

  static List<double> _toDoubleList(dynamic raw) {
    if (raw is! List) return const [];
    return raw.map((e) => (e as num).toDouble()).toList();
  }

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    final probs = <String, double>{};
    (json['class_probs'] as Map?)?.forEach((k, v) {
      probs['$k'] = (v as num).toDouble();
    });

    final colors = <String, Color>{};
    (json['class_colors'] as Map?)?.forEach((k, v) {
      colors['$k'] = _parseColor('$v');
    });

    final names = <String, String>{};
    (json['class_names'] as Map?)?.forEach((k, v) {
      names['$k'] = '$v';
    });

    final leads = <String, List<double>>{};
    (json['all_leads'] as Map?)?.forEach((k, v) {
      leads['$k'] = _toDoubleList(v);
    });

    return PredictionResult(
      prediction: '${json['prediction'] ?? '—'}',
      fullName: '${json['full_name'] ?? json['prediction'] ?? '—'}',
      description: '${json['description'] ?? ''}',
      keyFeatures:
          (json['key_features'] as List?)?.map((e) => '$e').toList() ?? const [],
      color: _parseColor('${json['color']}'),
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      classProbs: probs,
      classColors: colors,
      classNames: names,
      allLeads: leads,
      gradcam: _toDoubleList(json['gradcam']),
      gradcamLead: '${json['gradcam_lead'] ?? 'II'}',
      filename: '${json['filename'] ?? ''}',
      warning: json['warning'] as String?,
    );
  }
}
