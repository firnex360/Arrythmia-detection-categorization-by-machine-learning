import 'package:flutter/material.dart';

import 'api_service.dart';
import 'models.dart';
import 'patient_detail_screen.dart';
import 'theme.dart';

/// Dashboard de Riesgo y Alertas: patients prioritised by the model's latest
/// finding, recent abnormal cases, and abnormal ECGs still pending the doctor's
/// confirmation.
class RiskScreen extends StatefulWidget {
  const RiskScreen({super.key});

  @override
  State<RiskScreen> createState() => _RiskScreenState();
}

class _RiskScreenState extends State<RiskScreen> {
  RiskOverview? _data;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final d = await ApiService.risk();
      if (!mounted) return;
      setState(() => _data = d);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _openPatient(int id) async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => PatientDetailScreen(patientId: id)),
    );
    _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(

      body: RefreshIndicator(onRefresh: _load, child: _body()),
    );
  }

  Widget _body() {
    if (_error != null) {
      return ListView(children: [
        SizedBox(height: 80),
        Icon(Icons.cloud_off_rounded, size: 48, color: AppColors.muted),
        SizedBox(height: 12),
        Padding(
          padding: EdgeInsets.symmetric(horizontal: 32),
          child: Text(_error!,
              textAlign: TextAlign.center,
              style: TextStyle(color: AppColors.muted)),
        ),
        SizedBox(height: 16),
        Center(
          child: OutlinedButton.icon(
              onPressed: _load,
              icon: Icon(Icons.refresh),
              label: Text('Reintentar')),
        ),
      ]);
    }
    final d = _data;
    if (d == null) return Center(child: CircularProgressIndicator());

    if (d.prioritized.isEmpty) {
      return ListView(children: [
        SizedBox(height: 90),
        Icon(Icons.shield_outlined, size: 54, color: AppColors.muted),
        SizedBox(height: 14),
        Padding(
          padding: EdgeInsets.symmetric(horizontal: 32),
          child: Text(
            'Aún no tienes pacientes con ECG.\n'
            'Cuando agregues estudios, aquí verás quién necesita atención '
            'prioritaria según el modelo.',
            textAlign: TextAlign.center,
            style: TextStyle(color: AppColors.muted, height: 1.5),
          ),
        ),
      ]);
    }

    final prioritizedPanel = _Panel(
      title: 'Pacientes priorizados',
      subtitle: 'Ordenados por el riesgo de su ECG más reciente.',
      child: Column(
        children: [
          for (final p in d.prioritized)
            _RiskPatientTile(
              patient: p,
              classNames: d.classNames,
              color: d.classColors[p.latestPrediction ?? ''] ??
                  arrhythmiaColor(p.latestPrediction ?? ''),
              onTap: () => _openPatient(p.patientId),
            ),
        ],
      ),
    );

    final alertsColumn = Column(
      children: [
        _Panel(
          title: 'Seguimiento pendiente',
          subtitle: 'ECG anormales que aún no has confirmado.',
          child: d.pendingFollowup.isEmpty
              ? const _Empty('Nada pendiente. ¡Al día!')
              : Column(
                  children: [
                    for (final a in d.pendingFollowup)
                      _AlertTile(
                        alert: a,
                        pending: true,
                        color: d.classColors[a.prediction] ??
                            arrhythmiaColor(a.prediction),
                        onTap: () => _openPatient(a.patientId),
                      ),
                  ],
                ),
        ),
        SizedBox(height: 16),
        _Panel(
          title: 'Nuevos casos anormales',
          subtitle: 'Los ECG anormales más recientes.',
          child: d.newAbnormal.isEmpty
              ? const _Empty('Sin casos anormales registrados.')
              : Column(
                  children: [
                    for (final a in d.newAbnormal)
                      _AlertTile(
                        alert: a,
                        pending: false,
                        color: d.classColors[a.prediction] ??
                            arrhythmiaColor(a.prediction),
                        onTap: () => _openPatient(a.patientId),
                      ),
                  ],
                ),
        ),
      ],
    );

    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final wide = w >= 900;
        return ListView(
          padding: EdgeInsets.symmetric(
              horizontal: w >= 900 ? 32 : 16, vertical: 18),
          children: [
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1200),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _RiskCounts(counts: d.counts),
                    SizedBox(height: 18),
                    if (wide)
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(flex: 5, child: prioritizedPanel),
                          SizedBox(width: 16),
                          Expanded(flex: 4, child: alertsColumn),
                        ],
                      )
                    else ...[
                      prioritizedPanel,
                      SizedBox(height: 16),
                      alertsColumn,
                    ],
                  ],
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

Color riskColor(String level) => switch (level) {
      'alto' => const Color(0xFFEF4444),
      'medio' => const Color(0xFFF59E0B),
      'bajo' => const Color(0xFF3B82F6),
      _ => const Color(0xFF10B981),
    };

String riskLabel(String level) => switch (level) {
      'alto' => 'Alto',
      'medio' => 'Medio',
      'bajo' => 'Bajo',
      _ => 'Normal',
    };

class _RiskCounts extends StatelessWidget {
  final Map<String, int> counts;
  const _RiskCounts({required this.counts});

  @override
  Widget build(BuildContext context) {
    Widget tile(String level, IconData icon) {
      final c = riskColor(level);
      return Container(
        width: 150,
        padding: EdgeInsets.symmetric(vertical: 14, horizontal: 14),
        decoration: BoxDecoration(
          color: c.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: c.withValues(alpha: 0.4)),
        ),
        child: Row(
          children: [
            Icon(icon, color: c, size: 24),
            SizedBox(width: 10),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${counts[level] ?? 0}',
                    style: TextStyle(
                        color: c, fontSize: 22, fontWeight: FontWeight.w800)),
                Text('Riesgo ${riskLabel(level).toLowerCase()}',
                    style: TextStyle(color: AppColors.muted, fontSize: 11)),
              ],
            ),
          ],
        ),
      );
    }

    return Wrap(
      spacing: 12,
      runSpacing: 12,
      children: [
        tile('alto', Icons.priority_high_rounded),
        tile('medio', Icons.warning_amber_rounded),
        tile('bajo', Icons.info_outline_rounded),
        tile('normal', Icons.check_circle_outline),
      ],
    );
  }
}

class _RiskPatientTile extends StatelessWidget {
  final RiskPatient patient;
  final Map<String, String> classNames;
  final Color color;
  final VoidCallback onTap;

  const _RiskPatientTile({
    required this.patient,
    required this.classNames,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final level = patient.riskLevel;
    final lc = riskColor(level);
    final pred = patient.latestPrediction;
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 5),
      child: Material(
        color: AppColors.surface2,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onTap,
          child: Padding(
            padding: EdgeInsets.all(12),
            child: Row(
              children: [
                Container(
                  width: 6,
                  height: 42,
                  decoration: BoxDecoration(
                      color: lc, borderRadius: BorderRadius.circular(3)),
                ),
                SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Flexible(
                            child: Text(patient.name,
                                overflow: TextOverflow.ellipsis,
                                style: TextStyle(
                                    color: AppColors.text,
                                    fontWeight: FontWeight.w700,
                                    fontSize: 14.5)),
                          ),
                          SizedBox(width: 8),
                          _Badge(text: 'Riesgo ${riskLabel(level)}', color: lc),
                        ],
                      ),
                      SizedBox(height: 3),
                      Text(
                        [
                          if (patient.age != null) '${patient.age} años',
                          if (pred != null)
                            '$pred · ${(patient.latestConfidence * 100).toStringAsFixed(0)}%'
                          else
                            'Sin ECG',
                          '${patient.abnormalCount}/${patient.totalEcgs} anormales',
                        ].join(' · '),
                        style:
                            TextStyle(color: AppColors.muted, fontSize: 12),
                      ),
                    ],
                  ),
                ),
                if (patient.pendingReview)
                  Padding(
                    padding: EdgeInsets.only(left: 8),
                    child: Icon(Icons.rule_rounded,
                        size: 18, color: Color(0xFFF59E0B)),
                  ),
                Icon(Icons.chevron_right, color: AppColors.muted),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _AlertTile extends StatelessWidget {
  final AbnormalEcg alert;
  final bool pending;
  final Color color;
  final VoidCallback onTap;

  const _AlertTile({
    required this.alert,
    required this.pending,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: ListTile(
        dense: true,
        contentPadding: EdgeInsets.symmetric(horizontal: 8),
        leading: Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        title: Text(alert.name,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
                color: AppColors.text,
                fontWeight: FontWeight.w600,
                fontSize: 13.5)),
        subtitle: Text(
          '${alert.prediction} · ${(alert.confidence * 100).toStringAsFixed(0)}% · '
          '${_fmtDate(alert.createdAt)}',
          style: TextStyle(color: AppColors.muted, fontSize: 11.5),
        ),
        trailing: pending
            ? _Badge(text: 'Pendiente', color: const Color(0xFFF59E0B))
            : Icon(Icons.chevron_right, color: AppColors.muted),
        onTap: onTap,
      ),
    );
  }

  static String _fmtDate(String iso) {
    final dt = DateTime.tryParse(iso);
    if (dt == null) return iso;
    final l = dt.toLocal();
    String two(int n) => n.toString().padLeft(2, '0');
    return '${l.year}-${two(l.month)}-${two(l.day)} ${two(l.hour)}:${two(l.minute)}';
  }
}

class _Badge extends StatelessWidget {
  final String text;
  final Color color;
  const _Badge({required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.16),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Text(text,
          style: TextStyle(
              color: color, fontSize: 10.5, fontWeight: FontWeight.w700)),
    );
  }
}

class _Panel extends StatelessWidget {
  final String title;
  final String subtitle;
  final Widget child;
  const _Panel(
      {required this.title, required this.subtitle, required this.child});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.text)),
            SizedBox(height: 4),
            Text(subtitle,
                style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
            SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}

class _Empty extends StatelessWidget {
  final String text;
  const _Empty(this.text);
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 10),
      child: Row(
        children: [
          Icon(Icons.check_circle_outline,
              size: 18, color: AppColors.muted),
          SizedBox(width: 8),
          Expanded(
            child: Text(text,
                style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
          ),
        ],
      ),
    );
  }
}
