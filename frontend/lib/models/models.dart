import 'package:flutter/material.dart';

import 'package:frontend/core/theme.dart';

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

  static Color parseColor(String? hex, [Color fallback = const Color(0xFF38BDF8)]) =>
      _parseColor(hex, fallback);
}

// ══════════════════════════════════════════════════════════════════════════════
//  Clinical records — doctors, patients, ECG history, dashboard
// ══════════════════════════════════════════════════════════════════════════════

/// The logged-in doctor.
class Doctor {
  final int id;
  final String username;
  final String name;
  final String? avatarColor; // hex like '#38bdf8'
  Doctor({
    required this.id,
    required this.username,
    required this.name,
    this.avatarColor,
  });

  factory Doctor.fromJson(Map<String, dynamic> j) => Doctor(
        id: (j['id'] as num).toInt(),
        username: '${j['username'] ?? ''}',
        name: '${j['name'] ?? j['username'] ?? ''}',
        avatarColor: j['avatar_color'] as String?,
      );

  /// Up to two initials for the avatar circle.
  String get initials {
    final parts = name.trim().split(RegExp(r'\s+')).where((s) => s.isNotEmpty);
    if (parts.isEmpty) return '?';
    return parts.take(2).map((s) => s[0].toUpperCase()).join();
  }

  Color get color =>
      PredictionResult.parseColor(avatarColor, AppColors.accent);
}

/// A patient owned by the current doctor.
class Patient {
  final int id;
  final String? cedula;
  final String? firstName;
  final String? lastName;
  final String name; // full name (computed by backend)
  final String? dob; // ISO YYYY-MM-DD
  final int? age; // computed by the backend from dob
  final String? gender; // 'F' | 'M' | 'Other'
  final String? notes;
  final int recordCount;

  Patient({
    required this.id,
    this.cedula,
    this.firstName,
    this.lastName,
    required this.name,
    this.dob,
    this.age,
    this.gender,
    this.notes,
    this.recordCount = 0,
  });

  factory Patient.fromJson(Map<String, dynamic> j) => Patient(
        id: (j['id'] as num).toInt(),
        cedula: j['cedula'] as String?,
        firstName: j['first_name'] as String?,
        lastName: j['last_name'] as String?,
        name: '${j['name'] ?? ''}',
        dob: j['dob'] as String?,
        age: (j['age'] as num?)?.toInt(),
        gender: j['gender'] as String?,
        notes: j['notes'] as String?,
        recordCount: (j['record_count'] as num?)?.toInt() ?? 0,
      );

  String get genderLabel => switch (gender) {
        'F' => 'Femenino',
        'M' => 'Masculino',
        'Other' => 'Otro',
        _ => '—',
      };

  String get initials {
    final parts = name.trim().split(RegExp(r'\s+')).where((s) => s.isNotEmpty);
    if (parts.isEmpty) return '?';
    return parts.take(2).map((s) => s[0].toUpperCase()).join();
  }
}

/// One ECG analysis stored against a patient.
class EcgRecord {
  final int id;
  final int patientId;
  final String filename;
  final String prediction;
  final double confidence;
  final Map<String, double> classProbs;
  final String? doctorNotes;
  final String createdAt;

  /// Doctor's assessment: 'correct' | 'incorrect' | null (unreviewed).
  final String? verdict;

  /// Actual class when the doctor marked the prediction incorrect.
  final String? trueLabel;

  /// Full prediction payload (Grad-CAM, leads, description …) — present when the
  /// record was fetched in detail via `/records/<id>`.
  final PredictionResult? result;

  EcgRecord({
    required this.id,
    required this.patientId,
    required this.filename,
    required this.prediction,
    required this.confidence,
    required this.classProbs,
    required this.doctorNotes,
    required this.createdAt,
    this.verdict,
    this.trueLabel,
    this.result,
  });

  factory EcgRecord.fromJson(Map<String, dynamic> j) {
    final probs = <String, double>{};
    (j['class_probs'] as Map?)?.forEach((k, v) {
      probs['$k'] = (v as num).toDouble();
    });
    final full = j['result'];
    return EcgRecord(
      id: (j['id'] as num).toInt(),
      patientId: (j['patient_id'] as num).toInt(),
      filename: '${j['filename'] ?? ''}',
      prediction: '${j['prediction'] ?? '—'}',
      confidence: (j['confidence'] as num?)?.toDouble() ?? 0.0,
      classProbs: probs,
      doctorNotes: j['doctor_notes'] as String?,
      verdict: j['verdict'] as String?,
      trueLabel: j['true_label'] as String?,
      createdAt: '${j['created_at'] ?? ''}',
      result: (full is Map<String, dynamic> && full.isNotEmpty)
          ? PredictionResult.fromJson(full)
          : null,
    );
  }
}

/// Aggregate stats for the whole program (dashboard).
class DashboardData {
  final int totalDoctors;
  final int totalPatients;
  final int totalRecords;
  final List<String> classOrder;
  final Map<String, String> classNames;
  final Map<String, Color> classColors;
  final Map<String, int> byClass;
  final Map<String, double> avgConfidence;
  final List<GroupCounts> byGender;
  final List<GroupCounts> byAge;
  final AccuracyStats accuracy;
  final List<TimelinePoint> timelineDay;
  final List<TimelinePoint> timelineHour;

  DashboardData({
    required this.totalDoctors,
    required this.totalPatients,
    required this.totalRecords,
    required this.classOrder,
    required this.classNames,
    required this.classColors,
    required this.byClass,
    required this.avgConfidence,
    required this.byGender,
    required this.byAge,
    required this.accuracy,
    required this.timelineDay,
    required this.timelineHour,
  });

  factory DashboardData.fromJson(Map<String, dynamic> j) {
    final totals = (j['totals'] as Map?) ?? {};
    final order = (j['class_order'] as List?)?.map((e) => '$e').toList() ?? const [];

    final names = <String, String>{};
    (j['class_names'] as Map?)?.forEach((k, v) => names['$k'] = '$v');

    final colors = <String, Color>{};
    (j['class_colors'] as Map?)?.forEach(
        (k, v) => colors['$k'] = PredictionResult.parseColor('$v'));

    final byClass = <String, int>{};
    (j['by_class'] as Map?)?.forEach((k, v) => byClass['$k'] = (v as num).toInt());

    final avg = <String, double>{};
    (j['avg_confidence'] as Map?)?.forEach((k, v) => avg['$k'] = (v as num).toDouble());

    List<GroupCounts> parseGroups(dynamic raw, String labelKey) {
      if (raw is! List) return const [];
      return raw.map((e) {
        final m = e as Map<String, dynamic>;
        final counts = <String, int>{};
        (m['counts'] as Map?)?.forEach((k, v) => counts['$k'] = (v as num).toInt());
        return GroupCounts(
          label: '${m[labelKey] ?? ''}',
          counts: counts,
          total: (m['total'] as num?)?.toInt() ?? 0,
        );
      }).toList();
    }

    return DashboardData(
      totalDoctors: (totals['doctors'] as num?)?.toInt() ?? 0,
      totalPatients: (totals['patients'] as num?)?.toInt() ?? 0,
      totalRecords: (totals['records'] as num?)?.toInt() ?? 0,
      classOrder: order,
      classNames: names,
      classColors: colors,
      byClass: byClass,
      avgConfidence: avg,
      byGender: parseGroups(j['by_gender'], 'gender'),
      byAge: parseGroups(j['by_age'], 'group'),
      accuracy: AccuracyStats.fromJson(
          (j['accuracy'] as Map?)?.cast<String, dynamic>() ?? const {}),
      timelineDay: _parseTimeline(j, 'day'),
      timelineHour: _parseTimeline(j, 'hour'),
    );
  }

  static List<TimelinePoint> _parseTimeline(Map<String, dynamic> j, String key) {
    final tls = j['timelines'];
    final raw = (tls is Map && tls[key] is List)
        ? tls[key] as List
        : (key == 'day' ? (j['timeline'] as List? ?? const []) : const []);
    return raw
        .map((e) => TimelinePoint.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}

/// A named group (a gender or an age band) with per-class counts.
class GroupCounts {
  final String label;
  final Map<String, int> counts;
  final int total;
  GroupCounts({required this.label, required this.counts, required this.total});
}

/// One day on the timeline: how many ECGs were analysed, split by class.
class TimelinePoint {
  final String date; // YYYY-MM-DD
  final int total;
  final Map<String, int> counts;
  TimelinePoint({required this.date, required this.total, required this.counts});

  factory TimelinePoint.fromJson(Map<String, dynamic> j) {
    final counts = <String, int>{};
    (j['counts'] as Map?)?.forEach((k, v) => counts['$k'] = (v as num).toInt());
    return TimelinePoint(
      date: '${j['date'] ?? ''}',
      total: (j['total'] as num?)?.toInt() ?? 0,
      counts: counts,
    );
  }
}

/// How well the model does, according to the doctors' verdicts.
class AccuracyStats {
  final int reviewed;
  final int correct;
  final int unreviewed;
  final double? accuracy; // null when nothing reviewed yet
  final Map<String, ClassAccuracy> byClass;
  final Map<String, int> confusion; // 'PRED→ACTUAL' -> count

  AccuracyStats({
    required this.reviewed,
    required this.correct,
    required this.unreviewed,
    required this.accuracy,
    required this.byClass,
    required this.confusion,
  });

  factory AccuracyStats.fromJson(Map<String, dynamic> j) {
    final overall = (j['overall'] as Map?)?.cast<String, dynamic>() ?? const {};
    final byClass = <String, ClassAccuracy>{};
    (j['by_class'] as Map?)?.forEach((k, v) {
      byClass['$k'] = ClassAccuracy.fromJson((v as Map).cast<String, dynamic>());
    });
    final confusion = <String, int>{};
    (j['confusion'] as Map?)?.forEach((k, v) => confusion['$k'] = (v as num).toInt());
    return AccuracyStats(
      reviewed: (overall['reviewed'] as num?)?.toInt() ?? 0,
      correct: (overall['correct'] as num?)?.toInt() ?? 0,
      unreviewed: (overall['unreviewed'] as num?)?.toInt() ?? 0,
      accuracy: (overall['accuracy'] as num?)?.toDouble(),
      byClass: byClass,
      confusion: confusion,
    );
  }
}

class ClassAccuracy {
  final int reviewed;
  final int correct;
  final double? accuracy;
  ClassAccuracy(
      {required this.reviewed, required this.correct, required this.accuracy});

  factory ClassAccuracy.fromJson(Map<String, dynamic> j) => ClassAccuracy(
        reviewed: (j['reviewed'] as num?)?.toInt() ?? 0,
        correct: (j['correct'] as num?)?.toInt() ?? 0,
        accuracy: (j['accuracy'] as num?)?.toDouble(),
      );
}

// ══════════════════════════════════════════════════════════════════════════════
//  Risk & alerts
// ══════════════════════════════════════════════════════════════════════════════

/// Triage overview for the doctor's own patients.
class RiskOverview {
  final Map<String, int> counts; // level -> count (alto/medio/bajo/normal)
  final List<RiskPatient> prioritized;
  final List<AbnormalEcg> newAbnormal;
  final List<AbnormalEcg> pendingFollowup;
  final Map<String, String> classNames;
  final Map<String, Color> classColors;

  RiskOverview({
    required this.counts,
    required this.prioritized,
    required this.newAbnormal,
    required this.pendingFollowup,
    required this.classNames,
    required this.classColors,
  });

  factory RiskOverview.fromJson(Map<String, dynamic> j) {
    final counts = <String, int>{};
    (j['counts'] as Map?)?.forEach((k, v) => counts['$k'] = (v as num).toInt());
    final names = <String, String>{};
    (j['class_names'] as Map?)?.forEach((k, v) => names['$k'] = '$v');
    final colors = <String, Color>{};
    (j['class_colors'] as Map?)?.forEach(
        (k, v) => colors['$k'] = PredictionResult.parseColor('$v'));
    List<T> parse<T>(String key, T Function(Map<String, dynamic>) f) =>
        ((j[key] as List?) ?? const [])
            .map((e) => f(e as Map<String, dynamic>))
            .toList();
    return RiskOverview(
      counts: counts,
      prioritized: parse('prioritized', RiskPatient.fromJson),
      newAbnormal: parse('new_abnormal', AbnormalEcg.fromJson),
      pendingFollowup: parse('pending_followup', AbnormalEcg.fromJson),
      classNames: names,
      classColors: colors,
    );
  }
}

/// A patient ranked by risk (from their latest ECG).
class RiskPatient {
  final int patientId;
  final String name;
  final int? age;
  final String? gender;
  final int totalEcgs;
  final int abnormalCount;
  final String? latestPrediction;
  final double latestConfidence;
  final String? latestDate;
  final bool pendingReview;
  final double riskScore;
  final String riskLevel; // 'alto' | 'medio' | 'bajo' | 'normal'

  RiskPatient({
    required this.patientId,
    required this.name,
    required this.age,
    required this.gender,
    required this.totalEcgs,
    required this.abnormalCount,
    required this.latestPrediction,
    required this.latestConfidence,
    required this.latestDate,
    required this.pendingReview,
    required this.riskScore,
    required this.riskLevel,
  });

  factory RiskPatient.fromJson(Map<String, dynamic> j) => RiskPatient(
        patientId: (j['patient_id'] as num).toInt(),
        name: '${j['name'] ?? ''}',
        age: (j['age'] as num?)?.toInt(),
        gender: j['gender'] as String?,
        totalEcgs: (j['total_ecgs'] as num?)?.toInt() ?? 0,
        abnormalCount: (j['abnormal_count'] as num?)?.toInt() ?? 0,
        latestPrediction: j['latest_prediction'] as String?,
        latestConfidence: (j['latest_confidence'] as num?)?.toDouble() ?? 0.0,
        latestDate: j['latest_date'] as String?,
        pendingReview: j['pending_review'] == true,
        riskScore: (j['risk_score'] as num?)?.toDouble() ?? 0.0,
        riskLevel: '${j['risk_level'] ?? 'normal'}',
      );
}

/// A recent abnormal ECG for the alerts lists.
class AbnormalEcg {
  final int recordId;
  final int patientId;
  final String name;
  final String prediction;
  final double confidence;
  final String createdAt;
  final String? verdict;

  AbnormalEcg({
    required this.recordId,
    required this.patientId,
    required this.name,
    required this.prediction,
    required this.confidence,
    required this.createdAt,
    required this.verdict,
  });

  factory AbnormalEcg.fromJson(Map<String, dynamic> j) => AbnormalEcg(
        recordId: (j['record_id'] as num).toInt(),
        patientId: (j['patient_id'] as num).toInt(),
        name: '${j['name'] ?? ''}',
        prediction: '${j['prediction'] ?? '—'}',
        confidence: (j['confidence'] as num?)?.toDouble() ?? 0.0,
        createdAt: '${j['created_at'] ?? ''}',
        verdict: j['verdict'] as String?,
      );
}
