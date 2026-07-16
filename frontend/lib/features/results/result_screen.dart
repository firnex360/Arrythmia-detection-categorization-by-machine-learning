import 'package:flutter/material.dart';

import 'package:frontend/services/api_service.dart';
import 'package:frontend/models/models.dart';
import 'package:frontend/core/theme.dart';
import 'package:frontend/widgets/gradcam_ecg.dart';
import 'package:frontend/widgets/probability_bars.dart';

/// Full breakdown of a single prediction.
///
/// When [recordId] is provided (i.e. the result is a stored patient ECG), the
/// doctor can also add recommendations / notes, which are saved to the backend.
class ResultScreen extends StatefulWidget {
  final PredictionResult result;
  final int? recordId;
  final String? initialNotes;
  final String? initialVerdict; // 'correct' | 'incorrect' | null
  final String? initialTrueLabel;
  final bool alreadyExisted;
  final String? patientName;

  const ResultScreen({
    super.key,
    required this.result,
    this.recordId,
    this.initialNotes,
    this.initialVerdict,
    this.initialTrueLabel,
    this.alreadyExisted = false,
    this.patientName,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  late final TextEditingController _notes;
  bool _savingNotes = false;
  String? _savedNotes; // last value confirmed saved

  String? _verdict; // 'correct' | 'incorrect' | null
  String? _trueLabel; // actual class when incorrect
  bool _savingVerdict = false;

  PredictionResult get result => widget.result;

  @override
  void initState() {
    super.initState();
    _notes = TextEditingController(text: widget.initialNotes ?? '');
    _savedNotes = widget.initialNotes ?? '';
    _verdict = widget.initialVerdict;
    _trueLabel = widget.initialTrueLabel;
  }

  Future<void> _setVerdict(String? verdict, {String? trueLabel}) async {
    if (widget.recordId == null) return;
    setState(() => _savingVerdict = true);
    try {
      final updated = await ApiService.setRecordVerdict(
        widget.recordId!,
        verdict: verdict,
        trueLabel: trueLabel,
      );
      if (!mounted) return;
      setState(() {
        _verdict = updated.verdict;
        _trueLabel = updated.trueLabel;
        _savingVerdict = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _savingVerdict = false);
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  @override
  void dispose() {
    _notes.dispose();
    super.dispose();
  }

  bool get _notesDirty => _notes.text.trim() != (_savedNotes ?? '').trim();

  Future<void> _saveNotes() async {
    if (widget.recordId == null) return;
    setState(() => _savingNotes = true);
    try {
      final updated =
          await ApiService.updateRecordNotes(widget.recordId!, _notes.text.trim());
      if (!mounted) return;
      setState(() {
        _savedNotes = updated.doctorNotes ?? '';
        _savingNotes = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Notas guardadas.')));
    } catch (e) {
      if (!mounted) return;
      setState(() => _savingNotes = false);
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final leadSignal = result.allLeads[result.gradcamLead] ??
        result.allLeads.values.firstOrNull ??
        const [];

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.patientName != null
            ? 'ECG · ${widget.patientName}'
            : 'Resultado del análisis'),
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 860),
          child: ListView(
        padding: EdgeInsets.fromLTRB(16, 8, 16, 40),
        children: [
          if (widget.alreadyExisted) ...[
            _InfoBanner(
              text: 'Este archivo ya se había analizado para este paciente; '
                  'se muestra el resultado guardado.',
            ),
            SizedBox(height: 12),
          ],
          _VerdictCard(result: result),
          if (result.warning != null) ...[
            SizedBox(height: 12),
            _WarningBanner(text: result.warning!),
          ],
          if (widget.recordId != null) ...[
            SizedBox(height: 16),
            _DoctorVerdictCard(
              result: result,
              verdict: _verdict,
              trueLabel: _trueLabel,
              saving: _savingVerdict,
              onSet: _setVerdict,
            ),
          ],
          if (widget.recordId != null) ...[
            SizedBox(height: 16),
            _NotesCard(
              controller: _notes,
              saving: _savingNotes,
              dirty: _notesDirty,
              onSave: _saveNotes,
              onChanged: () => setState(() {}),
            ),
          ],
          SizedBox(height: 16),
          _SectionCard(
            title: 'Dónde miró el modelo (Grad-CAM)',
            subtitle:
                'El trazado es el ECG. Los colores más cálidos marcan los '
                'momentos que más influyeron en el veredicto.',
            child: GradCamEcg(
              signal: leadSignal,
              importance: result.gradcam,
              leadLabel: result.gradcamLead,
            ),
          ),
          SizedBox(height: 16),
          _SectionCard(
            title: 'Probabilidades de todas las clases',
            subtitle: 'Qué otras arritmias consideró el modelo, y con qué fuerza.',
            child: ProbabilityBars(result: result),
          ),
          SizedBox(height: 16),
          _SectionCard(
            title: 'Por qué este resultado',
            subtitle:
                'Características propias de ${result.fullName} — en qué se basa '
                'la lectura de este ritmo.',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(result.description,
                    style: TextStyle(
                        color: AppColors.text, height: 1.5, fontSize: 14)),
                if (result.keyFeatures.isNotEmpty) ...[
                  SizedBox(height: 14),
                  for (final f in result.keyFeatures) _FeatureRow(text: f),
                ],
              ],
            ),
          ),
        ],
          ),
        ),
      ),
    );
  }
}

/// Lets the doctor confirm whether the model's prediction was right. This feeds
/// the accuracy dashboards. When marked incorrect, the doctor picks the actual
/// arrhythmia so the system can track which classes get confused.
class _DoctorVerdictCard extends StatelessWidget {
  final PredictionResult result;
  final String? verdict;
  final String? trueLabel;
  final bool saving;
  final Future<void> Function(String? verdict, {String? trueLabel}) onSet;

  const _DoctorVerdictCard({
    required this.result,
    required this.verdict,
    required this.trueLabel,
    required this.saving,
    required this.onSet,
  });

  @override
  Widget build(BuildContext context) {
    final isCorrect = verdict == 'correct';
    final isIncorrect = verdict == 'incorrect';
    // Candidate "actual" classes (everything except the prediction).
    final others = result.classNames.keys
        .where((c) => c != result.prediction)
        .toList()
      ..sort();

    return Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.fact_check_outlined,
                    size: 18, color: AppColors.accent),
                SizedBox(width: 8),
                Expanded(
                  child: Text('¿El modelo acertó?',
                      style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: AppColors.text)),
                ),
                if (saving)
                  SizedBox(
                      width: 16,
                      height: 16,
                      child:
                          CircularProgressIndicator(strokeWidth: 2)),
              ],
            ),
            SizedBox(height: 4),
            Text(
              'Confirma el diagnóstico del modelo (${result.prediction}). '
              'Tu respuesta alimenta las estadísticas de precisión del panel.',
              style: TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
            SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: _VerdictButton(
                    label: 'Correcto',
                    icon: Icons.check_circle_outline,
                    color: const Color(0xFF10B981),
                    selected: isCorrect,
                    onTap: saving
                        ? null
                        : () => onSet(isCorrect ? null : 'correct'),
                  ),
                ),
                SizedBox(width: 12),
                Expanded(
                  child: _VerdictButton(
                    label: 'Incorrecto',
                    icon: Icons.cancel_outlined,
                    color: const Color(0xFFEF4444),
                    selected: isIncorrect,
                    onTap: saving
                        ? null
                        : () => onSet(isIncorrect ? null : 'incorrect',
                            trueLabel: trueLabel),
                  ),
                ),
              ],
            ),
            if (isIncorrect) ...[
              SizedBox(height: 14),
              Text('¿Cuál era la arritmia real?',
                  style: TextStyle(
                      color: AppColors.text,
                      fontSize: 13,
                      fontWeight: FontWeight.w600)),
              SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final code in others)
                    ChoiceChip(
                      label: Text('$code · ${result.classNames[code] ?? code}'),
                      selected: trueLabel == code,
                      backgroundColor: AppColors.surface2,
                      selectedColor: const Color(0x3338BDF8),
                      labelStyle: TextStyle(
                        color: trueLabel == code
                            ? AppColors.accent
                            : AppColors.text,
                        fontSize: 12.5,
                      ),
                      side: BorderSide(color: AppColors.border),
                      onSelected: saving
                          ? null
                          : (_) => onSet('incorrect', trueLabel: code),
                    ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _VerdictButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final bool selected;
  final VoidCallback? onTap;

  const _VerdictButton({
    required this.label,
    required this.icon,
    required this.color,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: selected ? color.withValues(alpha: 0.18) : AppColors.surface2,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Container(
          padding: EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
                color: selected ? color : AppColors.border,
                width: selected ? 1.5 : 1),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon,
                  size: 19, color: selected ? color : AppColors.muted),
              SizedBox(width: 8),
              Text(label,
                  style: TextStyle(
                      color: selected ? color : AppColors.text,
                      fontWeight: FontWeight.w700,
                      fontSize: 14)),
            ],
          ),
        ),
      ),
    );
  }
}

class _NotesCard extends StatelessWidget {
  final TextEditingController controller;
  final bool saving;
  final bool dirty;
  final VoidCallback onSave;
  final VoidCallback onChanged;

  const _NotesCard({
    required this.controller,
    required this.saving,
    required this.dirty,
    required this.onSave,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.sticky_note_2_outlined,
                    size: 18, color: AppColors.accent),
                SizedBox(width: 8),
                Text('Recomendaciones / notas del doctor',
                    style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: AppColors.text)),
              ],
            ),
            SizedBox(height: 4),
            Text(
              'Agrega tu interpretación clínica o recomendaciones para este ECG. '
              'Se guardan junto al resultado del modelo.',
              style: TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
            SizedBox(height: 12),
            TextField(
              controller: controller,
              minLines: 3,
              maxLines: 8,
              onChanged: (_) => onChanged(),
              style: TextStyle(color: AppColors.text, fontSize: 14),
              decoration: InputDecoration(
                hintText: 'Escribe aquí…',
                filled: true,
                fillColor: AppColors.surface2,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: BorderSide(color: AppColors.border),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: BorderSide(color: AppColors.border),
                ),
              ),
            ),
            SizedBox(height: 12),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.icon(
                onPressed: (saving || !dirty) ? null : onSave,
                style: FilledButton.styleFrom(
                  backgroundColor: AppColors.accent,
                  foregroundColor: Colors.black,
                ),
                icon: saving
                    ? SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.black))
                    : Icon(Icons.save_outlined, size: 18),
                label: Text(dirty ? 'Guardar notas' : 'Guardado'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _InfoBanner extends StatelessWidget {
  final String text;
  const _InfoBanner({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0x2238BDF8),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0x5538BDF8)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, color: AppColors.accent, size: 20),
          SizedBox(width: 10),
          Expanded(
            child: Text(text,
                style: TextStyle(
                    color: AppColors.text, fontSize: 12.5, height: 1.4)),
          ),
        ],
      ),
    );
  }
}

class _VerdictCard extends StatelessWidget {
  final PredictionResult result;
  const _VerdictCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 14,
                  height: 14,
                  decoration: BoxDecoration(
                      color: result.color, shape: BoxShape.circle),
                ),
                SizedBox(width: 10),
                Text(
                  result.prediction,
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.w800,
                    color: result.color,
                    letterSpacing: -0.5,
                  ),
                ),
              ],
            ),
            SizedBox(height: 4),
            Text(result.fullName,
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: AppColors.text)),
            SizedBox(height: 16),
            Row(
              children: [
                Text('Confidence',
                    style: TextStyle(color: AppColors.muted)),
                Spacer(),
                Text('${(result.confidence * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                        color: result.color, fontWeight: FontWeight.w700)),
              ],
            ),
            SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                value: result.confidence.clamp(0.0, 1.0),
                minHeight: 10,
                backgroundColor: AppColors.surface2,
                valueColor: AlwaysStoppedAnimation(result.color),
              ),
            ),
            if (result.filename.isNotEmpty) ...[
              SizedBox(height: 14),
              Text(result.filename,
                  style: TextStyle(
                      color: AppColors.muted,
                      fontSize: 12,
                      fontFamily: 'monospace')),
            ],
          ],
        ),
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final Widget child;
  const _SectionCard(
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
            SizedBox(height: 16),
            child,
          ],
        ),
      ),
    );
  }
}

class _FeatureRow extends StatelessWidget {
  final String text;
  const _FeatureRow({required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: EdgeInsets.only(top: 3, right: 10),
            child: Icon(Icons.monitor_heart_outlined,
                size: 16, color: AppColors.accent),
          ),
          Expanded(
            child: Text(text,
                style: TextStyle(
                    color: AppColors.text, height: 1.4, fontSize: 13.5)),
          ),
        ],
      ),
    );
  }
}

class _WarningBanner extends StatelessWidget {
  final String text;
  const _WarningBanner({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0x22F59E0B),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0x55F59E0B)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.warning_amber_rounded,
              color: Color(0xFFF59E0B), size: 20),
          SizedBox(width: 10),
          Expanded(
            child: Text(text,
                style: TextStyle(
                    color: Color(0xFFFBBF24), fontSize: 12.5, height: 1.4)),
          ),
        ],
      ),
    );
  }
}

extension _FirstOrNull<E> on Iterable<E> {
  E? get firstOrNull => isEmpty ? null : first;
}
