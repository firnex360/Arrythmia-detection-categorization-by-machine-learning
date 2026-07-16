import 'package:flutter/material.dart';

import 'package:frontend/services/api_service.dart';
import 'package:frontend/features/analysis/ecg_picker.dart';
import 'package:frontend/features/patients/individual_widgets.dart';
import 'package:frontend/models/models.dart';
import 'package:frontend/features/patients/patient_form.dart';
import 'package:frontend/features/results/result_screen.dart';
import 'package:frontend/core/theme.dart';

/// A patient's profile: general info, a summary of their ECG history, and every
/// stored analysis (each openable, each annotatable by the doctor).
class PatientDetailScreen extends StatefulWidget {
  final int patientId;
  const PatientDetailScreen({super.key, required this.patientId});

  @override
  State<PatientDetailScreen> createState() => _PatientDetailScreenState();
}

class _PatientDetailScreenState extends State<PatientDetailScreen> {
  Patient? _patient;
  List<EcgRecord> _records = const [];
  String? _error;
  bool _analyzing = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final (patient, records) = await ApiService.getPatient(widget.patientId);
      if (!mounted) return;
      setState(() {
        _patient = patient;
        _records = records;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _editPatient() async {
    if (_patient == null) return;
    final updated = await showPatientForm(context, existing: _patient);
    if (updated != null) _load();
  }

  Future<void> _deletePatient() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: Text('Eliminar paciente'),
        content: Text(
            '¿Eliminar a ${_patient?.name ?? 'este paciente'} y todos sus ECG? '
            'Esta acción no se puede deshacer.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: Text('Cancelar')),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: const Color(0xFFEF4444)),
            onPressed: () => Navigator.pop(context, true),
            child: Text('Eliminar'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    try {
      await ApiService.deletePatient(widget.patientId);
      if (mounted) Navigator.pop(context);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('$e')));
      }
    }
  }

  Future<void> _addEcg() async {
    final picked = await pickEcgFile();
    if (picked == null || !mounted) return;

    setState(() => _analyzing = true);
    try {
      final (record, existed) = await ApiService.analyzeForPatient(
        patientId: widget.patientId,
        bytes: picked.bytes,
        filename: picked.name,
      );
      if (!mounted) return;
      setState(() => _analyzing = false);

      if (existed) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text('Este archivo ya se había analizado antes.')));
      }
      await _openRecord(record, alreadyExisted: existed);
      _load();
    } catch (e) {
      if (!mounted) return;
      setState(() => _analyzing = false);
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          backgroundColor: AppColors.surface,
          title: Text('No se pudo analizar'),
          content: Text('$e'),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context),
                child: Text('OK')),
          ],
        ),
      );
    }
  }

  Future<void> _openRecord(EcgRecord record, {bool alreadyExisted = false}) async {
    // The record from /analyze already carries the full result; history rows
    // don't, so fetch the full payload when needed.
    EcgRecord full = record;
    if (record.result == null) {
      try {
        full = await ApiService.getRecord(record.id);
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context)
              .showSnackBar(SnackBar(content: Text('$e')));
        }
        return;
      }
    }
    if (full.result == null || !mounted) return;

    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => ResultScreen(
        result: full.result!,
        recordId: full.id,
        initialNotes: full.doctorNotes,
        initialVerdict: full.verdict,
        initialTrueLabel: full.trueLabel,
        alreadyExisted: alreadyExisted,
        patientName: _patient?.name,
      ),
    ));
    _load(); // notes may have changed
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_patient?.name ?? 'Paciente'),
        actions: [
          if (_patient != null) ...[
            IconButton(
                tooltip: 'Editar',
                onPressed: _editPatient,
                icon: Icon(Icons.edit_outlined)),
            IconButton(
                tooltip: 'Eliminar',
                onPressed: _deletePatient,
                icon: Icon(Icons.delete_outline)),
          ],
        ],
      ),
      floatingActionButton: _patient == null
          ? null
          : FloatingActionButton.extended(
              onPressed: _analyzing ? null : _addEcg,
              backgroundColor: AppColors.accent,
              foregroundColor: Colors.black,
              icon: _analyzing
                  ? SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.black))
                  : Icon(Icons.add_chart_rounded),
              label: Text(_analyzing ? 'Analizando…' : 'Agregar ECG'),
            ),
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
    if (_patient == null) {
      return Center(child: CircularProgressIndicator());
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final wide = w >= 820;
        final info = _InfoCard(patient: _patient!);
        final summary = _HistorySummary(records: _records);

        final header = wide
            ? Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(flex: 5, child: info),
                  if (_records.isNotEmpty) ...[
                    SizedBox(width: 16),
                    Expanded(flex: 4, child: summary),
                  ],
                ],
              )
            : Column(
                children: [
                  info,
                  if (_records.isNotEmpty) ...[
                    SizedBox(height: 16),
                    summary,
                  ],
                ],
              );

        return ListView(
          padding: EdgeInsets.symmetric(
              horizontal: w >= 900 ? 32 : 16, vertical: 12),
          children: [
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1100),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    header,
                    if (_records.isNotEmpty) ...[
                      SizedBox(height: 16),
                      CurrentDiagnosisCard(
                        record: _records.first,
                        onExplain: () => _openRecord(_records.first),
                      ),
                      SizedBox(height: 16),
                      EvolutionCard(records: _records),
                      if (_records.length >= 2) ...[
                        SizedBox(height: 16),
                        ComparisonCard(records: _records),
                      ],
                    ],
                    SizedBox(height: 18),
                    Text('Historial de ECG (${_records.length})',
                        style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            color: AppColors.text)),
                    SizedBox(height: 10),
                    if (_records.isEmpty)
                      Padding(
                        padding: EdgeInsets.symmetric(vertical: 28),
                        child: Center(
                          child: Text('Sin ECG registrados.\nUsa “Agregar ECG”.',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                  color: AppColors.muted, height: 1.5)),
                        ),
                      )
                    else
                      for (final r in _records)
                        _RecordTile(record: r, onTap: () => _openRecord(r)),
                    SizedBox(height: 80),
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

class _InfoCard extends StatelessWidget {
  final Patient patient;
  const _InfoCard({required this.patient});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(patient.name,
                style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: AppColors.text)),
            SizedBox(height: 12),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                if (patient.cedula != null && patient.cedula!.isNotEmpty)
                  _Chip(icon: Icons.badge_outlined, label: 'CC ${patient.cedula}'),
                _Chip(
                    icon: Icons.cake_outlined,
                    label: patient.age != null
                        ? '${patient.age} años'
                        : 'Edad —'),
                _Chip(icon: Icons.wc_outlined, label: patient.genderLabel),
                if (patient.dob != null && patient.dob!.isNotEmpty)
                  _Chip(icon: Icons.event_outlined, label: patient.dob!),
              ],
            ),
            if (patient.notes != null && patient.notes!.trim().isNotEmpty) ...[
              SizedBox(height: 14),
              Text('Datos adicionales',
                  style: TextStyle(
                      color: AppColors.muted,
                      fontSize: 12,
                      fontWeight: FontWeight.w600)),
              SizedBox(height: 4),
              Text(patient.notes!,
                  style: TextStyle(
                      color: AppColors.text, height: 1.5, fontSize: 13.5)),
            ],
          ],
        ),
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final IconData icon;
  final String label;
  const _Chip({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(
        color: AppColors.surface2,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 15, color: AppColors.muted),
          SizedBox(width: 6),
          Text(label,
              style: TextStyle(color: AppColors.text, fontSize: 12.5)),
        ],
      ),
    );
  }
}

/// A quick read on the patient's ECG history: how many of each rhythm, plus the
/// most recent verdict.
class _HistorySummary extends StatelessWidget {
  final List<EcgRecord> records;
  const _HistorySummary({required this.records});

  @override
  Widget build(BuildContext context) {
    if (records.isEmpty) return SizedBox.shrink();

    final counts = <String, int>{};
    for (final r in records) {
      counts[r.prediction] = (counts[r.prediction] ?? 0) + 1;
    }
    Color colorFor(String code) => arrhythmiaColor(code);
    final entries = counts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final total = records.length;
    final latest = records.first; // list is newest-first

    return Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Resumen de ECG',
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.text)),
            SizedBox(height: 4),
            Text('Último resultado: ${latest.prediction} · '
                '${(latest.confidence * 100).toStringAsFixed(0)}% de confianza',
                style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
            SizedBox(height: 14),
            // Stacked distribution bar.
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Row(
                children: [
                  for (final e in entries)
                    Expanded(
                      flex: e.value,
                      child: Container(
                        height: 12,
                        color: colorFor(e.key),
                      ),
                    ),
                ],
              ),
            ),
            SizedBox(height: 12),
            Wrap(
              spacing: 14,
              runSpacing: 6,
              children: [
                for (final e in entries)
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: colorFor(e.key),
                          shape: BoxShape.circle,
                        ),
                      ),
                      SizedBox(width: 6),
                      Text('${e.key}  ${e.value}/$total',
                          style: TextStyle(
                              color: AppColors.text, fontSize: 12)),
                    ],
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _RecordTile extends StatelessWidget {
  final EcgRecord record;
  final VoidCallback onTap;
  const _RecordTile({required this.record, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final color = record.result?.color ?? arrhythmiaColor(record.prediction);
    final hasNote =
        record.doctorNotes != null && record.doctorNotes!.trim().isNotEmpty;
    final verdictIcon = switch (record.verdict) {
      'correct' => (Icons.check_circle, const Color(0xFF10B981)),
      'incorrect' => (Icons.cancel, const Color(0xFFEF4444)),
      _ => null,
    };
    return Card(
      margin: EdgeInsets.only(bottom: 10),
      child: ListTile(
        contentPadding: EdgeInsets.symmetric(horizontal: 14, vertical: 4),
        leading: Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        title: Row(
          children: [
            Text(record.prediction,
                style: TextStyle(
                    color: AppColors.text,
                    fontWeight: FontWeight.w700,
                    fontSize: 15)),
            SizedBox(width: 8),
            Text('${(record.confidence * 100).toStringAsFixed(0)}%',
                style: TextStyle(color: AppColors.muted, fontSize: 12)),
            if (hasNote) ...[
              SizedBox(width: 8),
              Icon(Icons.sticky_note_2_outlined,
                  size: 15, color: AppColors.accent),
            ],
            if (verdictIcon != null) ...[
              SizedBox(width: 8),
              Icon(verdictIcon.$1, size: 15, color: verdictIcon.$2),
            ],
          ],
        ),
        subtitle: Text(
          '${_fmtDate(record.createdAt)}${record.filename.isNotEmpty ? ' · ${record.filename}' : ''}',
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: TextStyle(color: AppColors.muted, fontSize: 12),
        ),
        trailing: Icon(Icons.chevron_right, color: AppColors.muted),
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
