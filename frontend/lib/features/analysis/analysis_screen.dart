import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/material.dart';

import 'package:frontend/core/theme.dart';
import 'package:frontend/features/analysis/ecg_picker.dart';
import 'package:frontend/features/results/result_screen.dart';
import 'package:frontend/services/api_service.dart';

/// The doctor's first action: import an ECG file and analyse it. The result is
/// shown WITHOUT a patient — the doctor reviews everything and can then assign it
/// to a patient (existing or new) from the result screen.
class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen>
    with ThemeReactive<AnalysisScreen> {
  PickedEcg? _file;
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
  }

  Future<void> _analyze() async {
    if (_file == null || _busy) return;
    setState(() => _busy = true);
    try {
      final result =
          await ApiService.predict(bytes: _file!.bytes, filename: _file!.name);
      if (!mounted) return;
      final source = _file!;
      setState(() {
        _busy = false;
        _file = null; // ready for the next one
      });
      await Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => ResultScreen(result: result, source: source),
      ));
    } catch (e) {
      if (!mounted) return;
      setState(() => _busy = false);
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          backgroundColor: AppColors.surface,
          title: const Text('No se pudo analizar'),
          content: Text('$e'),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK')),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 680),
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
                'Importa un archivo de ECG (.pt · .mat · .dat). Verás el resultado '
                'completo y luego podrás asignarlo a un paciente.',
                style: TextStyle(color: AppColors.muted, fontSize: 13),
              ),
              const SizedBox(height: 20),
              if (_error != null) ...[
                _ErrorBanner(text: _error!),
                const SizedBox(height: 16),
              ],
              if (_file == null) _dropZone() else _fileReady(),
            ],
          ),
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
            padding: const EdgeInsets.symmetric(vertical: 52, horizontal: 20),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
            ),
            child: Column(
              children: [
                Icon(Icons.cloud_upload_outlined,
                    size: 54, color: AppColors.accent),
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

  Widget _fileReady() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          padding: const EdgeInsets.all(16),
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
        const SizedBox(height: 16),
        FilledButton.icon(
          onPressed: _busy ? null : _analyze,
          style: FilledButton.styleFrom(
            minimumSize: const Size.fromHeight(50),
            backgroundColor: AppColors.accent,
            foregroundColor: Colors.black,
          ),
          icon: _busy
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: Colors.black))
              : const Icon(Icons.analytics_outlined),
          label: Text(_busy ? 'Analizando…' : 'Analizar ECG'),
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
