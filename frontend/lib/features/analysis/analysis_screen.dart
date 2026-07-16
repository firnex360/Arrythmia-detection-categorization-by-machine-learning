import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/material.dart';

import 'package:frontend/services/api_service.dart';
import 'package:frontend/features/analysis/ecg_picker.dart';
import 'package:frontend/models/models.dart';
import 'package:frontend/features/patients/patient_form.dart';
import 'package:frontend/features/results/result_screen.dart';
import 'package:frontend/core/theme.dart';

/// The doctor's first action: import an ECG file, then attach it to a patient
/// (existing or newly created) and run the analysis.
class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  PickedEcg? _file;
  List<Patient>? _patients;
  String _query = '';
  bool _busy = false;
  String? _error;

  Future<void> _pick() async {
    try {
      final f = await pickEcgFile();
      if (f == null) return;
      setState(() {
        _file = f;
        _error = null;
      });
      _loadPatients();
    } catch (e) {
      setState(() => _error = '$e');
    }
  }

  Future<void> _handleDrop(DropDoneDetails d) async {
    if (d.files.isEmpty) return;
    final xf = d.files.first;
    final name = xf.name.toLowerCase();
    if (!(name.endsWith('.pt') || name.endsWith('.mat') || name.endsWith('.dat'))) {
      setState(() => _error = 'Formato no soportado. Usa .pt, .mat o .dat.');
      return;
    }
    final bytes = await xf.readAsBytes();
    setState(() {
      _file = PickedEcg(bytes, xf.name);
      _error = null;
    });
    _loadPatients();
  }

  Future<void> _loadPatients() async {
    try {
      final list = await ApiService.listPatients();
      if (mounted) setState(() => _patients = list);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  Future<void> _analyzeFor(Patient p) async {
    if (_file == null || _busy) return;
    setState(() => _busy = true);
    try {
      final (record, existed) = await ApiService.analyzeForPatient(
        patientId: p.id,
        bytes: _file!.bytes,
        filename: _file!.name,
      );
      EcgRecord full = record;
      if (record.result == null) full = await ApiService.getRecord(record.id);
      if (!mounted) return;
      setState(() {
        _busy = false;
        _file = null; // reset for next analysis
        _query = '';
      });
      if (full.result != null) {
        await Navigator.of(context).push(MaterialPageRoute(
          builder: (_) => ResultScreen(
            result: full.result!,
            recordId: full.id,
            initialNotes: full.doctorNotes,
            initialVerdict: full.verdict,
            initialTrueLabel: full.trueLabel,
            alreadyExisted: existed,
            patientName: p.name,
          ),
        ));
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _busy = false);
      _showError('$e');
    }
  }

  Future<void> _createAndAnalyze() async {
    final created = await showPatientForm(context);
    if (created != null) {
      _loadPatients();
      _analyzeFor(created);
    }
  }

  void _showError(String msg) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: const Text('No se pudo analizar'),
        content: Text(msg),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context), child: const Text('OK')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 760),
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Text('Analizar un ECG',
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    color: AppColors.text)),
            const SizedBox(height: 4),
            Text(
              _file == null
                  ? 'Importa un archivo de ECG (.pt · .mat · .dat) para que el modelo lo analice.'
                  : 'Elige el paciente al que pertenece este ECG, o crea uno nuevo.',
              style: TextStyle(color: AppColors.muted, fontSize: 13),
            ),
            const SizedBox(height: 20),
            if (_error != null) ...[
              _ErrorBanner(text: _error!),
              const SizedBox(height: 16),
            ],
            if (_file == null) _dropZone() else _patientChooser(),
          ],
        ),
      ),
    );
  }

  Widget _dropZone() {
    return DropTarget(
      onDragDone: _handleDrop,
      child: Material(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: _pick,
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 20),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
            ),
            child: Column(
              children: [
                Icon(Icons.cloud_upload_outlined,
                    size: 52, color: AppColors.accent),
                const SizedBox(height: 14),
                Text('Haz clic para elegir un archivo o arrástralo aquí',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                        color: AppColors.text,
                        fontWeight: FontWeight.w700,
                        fontSize: 15)),
                const SizedBox(height: 6),
                Text('.pt · .mat · .dat',
                    style: TextStyle(color: AppColors.muted, fontSize: 12)),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _patientChooser() {
    final all = _patients;
    final items = (all ?? const <Patient>[]).where((p) {
      if (_query.trim().isEmpty) return true;
      final q = _query.toLowerCase();
      return p.name.toLowerCase().contains(q) ||
          (p.cedula ?? '').toLowerCase().contains(q);
    }).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Selected file chip.
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            children: [
              Icon(Icons.description_outlined, color: AppColors.accent),
              const SizedBox(width: 12),
              Expanded(
                child: Text(_file!.name,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                        color: AppColors.text, fontWeight: FontWeight.w600)),
              ),
              TextButton.icon(
                onPressed: _busy ? null : () => setState(() => _file = null),
                icon: const Icon(Icons.close, size: 16),
                label: const Text('Cambiar'),
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        Row(
          children: [
            Expanded(
              child: TextField(
                onChanged: (v) => setState(() => _query = v),
                style: TextStyle(color: AppColors.text),
                decoration: InputDecoration(
                  hintText: 'Buscar paciente por nombre o cédula…',
                  prefixIcon: const Icon(Icons.search, size: 20),
                  filled: true,
                  fillColor: AppColors.surface,
                  border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide(color: AppColors.border)),
                  enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide(color: AppColors.border)),
                ),
              ),
            ),
            const SizedBox(width: 10),
            FilledButton.icon(
              onPressed: _busy ? null : _createAndAnalyze,
              style: FilledButton.styleFrom(
                  backgroundColor: AppColors.accent,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18)),
              icon: const Icon(Icons.person_add_alt_1, size: 18),
              label: const Text('Nuevo'),
            ),
          ],
        ),
        const SizedBox(height: 14),
        if (_busy)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 30),
            child: Center(child: CircularProgressIndicator()),
          )
        else if (all == null)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 30),
            child: Center(child: CircularProgressIndicator()),
          )
        else if (items.isEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 24),
            child: Text(
              all.isEmpty
                  ? 'No tienes pacientes todavía. Crea uno con el botón “Nuevo”.'
                  : 'Ningún paciente coincide con la búsqueda.',
              style: TextStyle(color: AppColors.muted),
            ),
          )
        else
          for (final p in items)
            Card(
              margin: const EdgeInsets.only(bottom: 8),
              child: ListTile(
                leading: CircleAvatar(
                  backgroundColor: AppColors.accent.withValues(alpha: 0.15),
                  child: Text(p.initials,
                      style: TextStyle(
                          color: AppColors.accent, fontWeight: FontWeight.w700)),
                ),
                title: Text(p.name,
                    style: TextStyle(
                        color: AppColors.text, fontWeight: FontWeight.w600)),
                subtitle: Text(
                  [
                    if (p.cedula != null && p.cedula!.isNotEmpty) 'CC ${p.cedula}',
                    if (p.age != null) '${p.age} años',
                    p.genderLabel,
                  ].join(' · '),
                  style: TextStyle(color: AppColors.muted, fontSize: 12),
                ),
                trailing: Icon(Icons.play_arrow_rounded, color: AppColors.accent),
                onTap: () => _analyzeFor(p),
              ),
            ),
      ],
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String text;
  const _ErrorBanner({required this.text});
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0x22EF4444),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0x55EF4444)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Color(0xFFEF4444), size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text,
                style: const TextStyle(color: Color(0xFFFCA5A5), fontSize: 12.5)),
          ),
        ],
      ),
    );
  }
}
