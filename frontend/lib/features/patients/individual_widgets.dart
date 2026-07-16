import 'package:flutter/material.dart';

import 'package:frontend/models/models.dart';
import 'package:frontend/core/theme.dart';

// ══════════════════════════════════════════════════════════════════════════════
//  Individual clinical dashboard widgets (patient detail)
// ══════════════════════════════════════════════════════════════════════════════

String _fmtDateTime(String iso) {
  final dt = DateTime.tryParse(iso);
  if (dt == null) return iso;
  final l = dt.toLocal();
  String two(int n) => n.toString().padLeft(2, '0');
  return '${l.year}-${two(l.month)}-${two(l.day)} ${two(l.hour)}:${two(l.minute)}';
}

/// Current diagnosis (latest ECG) with a shortcut to the Grad-CAM explanation.
class CurrentDiagnosisCard extends StatelessWidget {
  final EcgRecord record;
  final VoidCallback onExplain;
  const CurrentDiagnosisCard(
      {super.key, required this.record, required this.onExplain});

  @override
  Widget build(BuildContext context) {
    final color = arrhythmiaColor(record.prediction);
    return Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.medical_information_outlined,
                    size: 18, color: AppColors.accent),
                SizedBox(width: 8),
                Text('Diagnóstico actual',
                    style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: AppColors.text)),
                Spacer(),
                Flexible(
                  child: Text(_fmtDateTime(record.createdAt),
                      overflow: TextOverflow.ellipsis,
                      style:
                          TextStyle(color: AppColors.muted, fontSize: 11)),
                ),
              ],
            ),
            SizedBox(height: 14),
            Row(
              children: [
                Container(
                  width: 14,
                  height: 14,
                  decoration:
                      BoxDecoration(color: color, shape: BoxShape.circle),
                ),
                SizedBox(width: 10),
                Text(record.prediction,
                    style: TextStyle(
                        color: color,
                        fontSize: 26,
                        fontWeight: FontWeight.w800)),
                Spacer(),
                Text('${(record.confidence * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                        color: color,
                        fontSize: 16,
                        fontWeight: FontWeight.w700)),
              ],
            ),
            SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(5),
              child: LinearProgressIndicator(
                value: record.confidence.clamp(0.0, 1.0),
                minHeight: 8,
                backgroundColor: AppColors.surface2,
                valueColor: AlwaysStoppedAnimation(color),
              ),
            ),
            if (record.doctorNotes != null &&
                record.doctorNotes!.trim().isNotEmpty) ...[
              SizedBox(height: 12),
              Text(record.doctorNotes!,
                  style: TextStyle(
                      color: AppColors.text, fontSize: 12.5, height: 1.4)),
            ],
            SizedBox(height: 14),
            Align(
              alignment: Alignment.centerLeft,
              child: FilledButton.tonalIcon(
                onPressed: onExplain,
                icon: Icon(Icons.insights_rounded, size: 18),
                label: Text('Ver explicación (Grad-CAM)'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Chronological strip of the patient's studies — ECG evolution over time.
class EvolutionCard extends StatelessWidget {
  final List<EcgRecord> records;
  const EvolutionCard({super.key, required this.records});

  @override
  Widget build(BuildContext context) {
    final ordered = records.reversed.toList(); // oldest -> newest
    return Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Evolución de los ECG',
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.text)),
            SizedBox(height: 4),
            Text(
                'Secuencia de estudios del más antiguo al más reciente. La altura '
                'de cada barra indica la confianza del modelo.',
                style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
            SizedBox(height: 16),
            SizedBox(
              height: 130,
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    for (var i = 0; i < ordered.length; i++) ...[
                      if (i > 0)
                        Padding(
                          padding: EdgeInsets.only(bottom: 44),
                          child: Icon(Icons.arrow_right_alt,
                              size: 18, color: AppColors.muted),
                        ),
                      _EvolutionMarker(record: ordered[i]),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _EvolutionMarker extends StatelessWidget {
  final EcgRecord record;
  const _EvolutionMarker({required this.record});

  @override
  Widget build(BuildContext context) {
    final color = arrhythmiaColor(record.prediction);
    final barH = (record.confidence.clamp(0.0, 1.0) * 58) + 6;
    return SizedBox(
      width: 76,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          Text('${(record.confidence * 100).toStringAsFixed(0)}%',
              style: TextStyle(color: AppColors.muted, fontSize: 10)),
          SizedBox(height: 4),
          Container(
            width: 22,
            height: barH,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(5),
            ),
          ),
          SizedBox(height: 6),
          Text(record.prediction,
              style: TextStyle(
                  color: color, fontWeight: FontWeight.w700, fontSize: 12)),
          Text(_shortDate(record.createdAt),
              style: TextStyle(color: AppColors.muted, fontSize: 10)),
        ],
      ),
    );
  }

  static String _shortDate(String iso) {
    final dt = DateTime.tryParse(iso);
    if (dt == null) return '';
    final l = dt.toLocal();
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(l.month)}-${two(l.day)} ${two(l.hour)}:${two(l.minute)}';
  }
}

/// Side-by-side comparison of any two of the patient's studies.
class ComparisonCard extends StatefulWidget {
  final List<EcgRecord> records;
  const ComparisonCard({super.key, required this.records});

  @override
  State<ComparisonCard> createState() => _ComparisonCardState();
}

class _ComparisonCardState extends State<ComparisonCard> {
  late int _aId;
  late int _bId;

  @override
  void initState() {
    super.initState();
    _bId = widget.records[0].id; // newest
    _aId = widget.records[1].id; // second newest
  }

  EcgRecord _byId(int id) => widget.records.firstWhere((r) => r.id == id);

  String _label(EcgRecord r) => '${r.prediction} · ${_fmtDateTime(r.createdAt)}';

  @override
  Widget build(BuildContext context) {
    final a = _byId(_aId);
    final b = _byId(_bId);
    final changed = a.prediction != b.prediction;

    return Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Comparación entre estudios',
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.text)),
            SizedBox(height: 4),
            Text(
                'Elige dos ECG para contrastar el diagnóstico y las probabilidades.',
                style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
            SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                    child: _selector(
                        'Estudio A', _aId, (v) => setState(() => _aId = v))),
                SizedBox(width: 12),
                Expanded(
                    child: _selector(
                        'Estudio B', _bId, (v) => setState(() => _bId = v))),
              ],
            ),
            SizedBox(height: 16),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: _StudyColumn(record: a)),
                SizedBox(width: 12),
                Expanded(child: _StudyColumn(record: b)),
              ],
            ),
            SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color:
                    changed ? const Color(0x22F59E0B) : const Color(0x2210B981),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                    color: changed
                        ? const Color(0x55F59E0B)
                        : const Color(0x5510B981)),
              ),
              child: Row(
                children: [
                  Icon(changed ? Icons.trending_up : Icons.check_circle_outline,
                      size: 18,
                      color: changed
                          ? const Color(0xFFF59E0B)
                          : const Color(0xFF10B981)),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      changed
                          ? 'El diagnóstico cambió: ${a.prediction} → ${b.prediction}.'
                          : 'Mismo diagnóstico en ambos estudios (${a.prediction}).',
                      style:
                          TextStyle(color: AppColors.text, fontSize: 12.5),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _selector(String label, int value, ValueChanged<int> onChanged) {
    return DropdownButtonFormField<int>(
      initialValue: value,
      isExpanded: true,
      dropdownColor: AppColors.surface2,
      style: TextStyle(color: AppColors.text, fontSize: 12.5),
      decoration: InputDecoration(
        labelText: label,
        isDense: true,
        border: OutlineInputBorder(),
      ),
      items: [
        for (final r in widget.records)
          DropdownMenuItem(
            value: r.id,
            child: Text(_label(r), overflow: TextOverflow.ellipsis),
          ),
      ],
      onChanged: (v) {
        if (v != null) onChanged(v);
      },
    );
  }
}

class _StudyColumn extends StatelessWidget {
  final EcgRecord record;
  const _StudyColumn({required this.record});

  @override
  Widget build(BuildContext context) {
    final color = arrhythmiaColor(record.prediction);
    final probs = record.classProbs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return Container(
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surface2,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(record.prediction,
              style: TextStyle(
                  color: color, fontWeight: FontWeight.w800, fontSize: 18)),
          Text('${(record.confidence * 100).toStringAsFixed(1)}% confianza',
              style: TextStyle(color: AppColors.muted, fontSize: 11)),
          SizedBox(height: 10),
          for (final e in probs) _miniBar(e.key, e.value),
        ],
      ),
    );
  }

  Widget _miniBar(String code, double p) {
    final color = arrhythmiaColor(code);
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 3),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(code,
                    style:
                        TextStyle(color: AppColors.text, fontSize: 11.5)),
              ),
              Text('${(p * 100).toStringAsFixed(0)}%',
                  style: TextStyle(color: AppColors.muted, fontSize: 11)),
            ],
          ),
          SizedBox(height: 3),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: p.clamp(0.0, 1.0),
              minHeight: 6,
              backgroundColor: AppColors.bg,
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
        ],
      ),
    );
  }
}
