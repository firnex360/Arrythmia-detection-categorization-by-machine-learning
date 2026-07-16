import 'package:flutter/material.dart';

import 'api_service.dart';
import 'models.dart';
import 'theme.dart';

/// Opens the add/edit patient sheet. Returns the created/updated [Patient], or
/// null if cancelled.
Future<Patient?> showPatientForm(BuildContext context, {Patient? existing}) {
  return showModalBottomSheet<Patient>(
    context: context,
    isScrollControlled: true,
    backgroundColor: AppColors.surface,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
    ),
    builder: (_) => Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      child: _PatientForm(existing: existing),
    ),
  );
}

class _PatientForm extends StatefulWidget {
  final Patient? existing;
  const _PatientForm({this.existing});

  @override
  State<_PatientForm> createState() => _PatientFormState();
}

class _PatientFormState extends State<_PatientForm> {
  late final TextEditingController _cedula;
  late final TextEditingController _first;
  late final TextEditingController _last;
  late final TextEditingController _notes;
  DateTime? _dob;
  String? _gender;
  bool _busy = false;
  String? _error;

  bool get _isEdit => widget.existing != null;

  @override
  void initState() {
    super.initState();
    final p = widget.existing;
    _cedula = TextEditingController(text: p?.cedula ?? '');
    // Fall back to splitting the full name if the split fields are empty.
    final fallback = (p?.name ?? '').trim().split(RegExp(r'\s+'));
    _first = TextEditingController(
        text: p?.firstName ?? (fallback.isNotEmpty ? fallback.first : ''));
    _last = TextEditingController(
        text: p?.lastName ??
            (fallback.length > 1 ? fallback.sublist(1).join(' ') : ''));
    _notes = TextEditingController(text: p?.notes ?? '');
    _gender = p?.gender;
    if (p?.dob != null && p!.dob!.isNotEmpty) _dob = DateTime.tryParse(p.dob!);
  }

  @override
  void dispose() {
    _cedula.dispose();
    _first.dispose();
    _last.dispose();
    _notes.dispose();
    super.dispose();
  }

  int? get _age {
    if (_dob == null) return null;
    final now = DateTime.now();
    var a = now.year - _dob!.year;
    if (now.month < _dob!.month ||
        (now.month == _dob!.month && now.day < _dob!.day)) {
      a--;
    }
    return a;
  }

  Future<void> _pickDob() async {
    final now = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: _dob ?? DateTime(now.year - 40),
      firstDate: DateTime(1900),
      lastDate: now,
      helpText: 'Fecha de nacimiento',
    );
    if (picked != null) setState(() => _dob = picked);
  }

  String? _dobIso() {
    if (_dob == null) return null;
    final m = _dob!.month.toString().padLeft(2, '0');
    final d = _dob!.day.toString().padLeft(2, '0');
    return '${_dob!.year}-$m-$d';
  }

  Future<void> _save() async {
    if (_first.text.trim().isEmpty && _last.text.trim().isEmpty) {
      setState(() => _error = 'Ingresa al menos nombres o apellidos.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final Patient result;
      if (_isEdit) {
        result = await ApiService.updatePatient(
          widget.existing!.id,
          cedula: _cedula.text.trim(),
          firstName: _first.text.trim(),
          lastName: _last.text.trim(),
          dob: _dobIso(),
          gender: _gender,
          notes: _notes.text.trim(),
        );
      } else {
        result = await ApiService.createPatient(
          cedula: _cedula.text.trim(),
          firstName: _first.text.trim(),
          lastName: _last.text.trim(),
          dob: _dobIso(),
          gender: _gender,
          notes: _notes.text.trim(),
        );
      }
      if (mounted) Navigator.pop(context, result);
    } catch (e) {
      if (mounted) {
        setState(() {
          _busy = false;
          _error = '$e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(_isEdit ? 'Editar paciente' : 'Nuevo paciente',
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    color: AppColors.text)),
            const SizedBox(height: 18),
            _field(_cedula, 'Cédula / documento', Icons.badge_outlined),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(child: _field(_first, 'Nombres', Icons.person_outline)),
                const SizedBox(width: 12),
                Expanded(child: _field(_last, 'Apellidos', Icons.people_outline)),
              ],
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: InkWell(
                    onTap: _pickDob,
                    borderRadius: BorderRadius.circular(4),
                    child: InputDecorator(
                      decoration: const InputDecoration(
                        labelText: 'Fecha de nacimiento',
                        prefixIcon: Icon(Icons.cake_outlined, size: 20),
                        border: OutlineInputBorder(),
                      ),
                      child: Text(_dobIso() ?? 'Seleccionar',
                          style: TextStyle(
                              color: _dob == null
                                  ? AppColors.muted
                                  : AppColors.text)),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 14, vertical: 16),
                  decoration: BoxDecoration(
                    border: Border.all(color: AppColors.border),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Column(
                    children: [
                      Text('Edad',
                          style:
                              TextStyle(color: AppColors.muted, fontSize: 11)),
                      const SizedBox(height: 2),
                      Text(_age?.toString() ?? '—',
                          style: TextStyle(
                              color: AppColors.accent,
                              fontWeight: FontWeight.w700,
                              fontSize: 16)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            DropdownButtonFormField<String>(
              initialValue: _gender,
              dropdownColor: AppColors.surface2,
              decoration: const InputDecoration(
                labelText: 'Género',
                prefixIcon: Icon(Icons.wc_outlined, size: 20),
                border: OutlineInputBorder(),
              ),
              items: const [
                DropdownMenuItem(value: 'F', child: Text('Femenino')),
                DropdownMenuItem(value: 'M', child: Text('Masculino')),
                DropdownMenuItem(value: 'Other', child: Text('Otro')),
              ],
              onChanged: (v) => setState(() => _gender = v),
            ),
            const SizedBox(height: 14),
            TextField(
              controller: _notes,
              minLines: 2,
              maxLines: 4,
              style: TextStyle(color: AppColors.text),
              decoration: const InputDecoration(
                labelText: 'Notas / datos relevantes',
                hintText: 'Antecedentes, medicación, alergias…',
                alignLabelWithHint: true,
                border: OutlineInputBorder(),
              ),
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Text(_error!,
                  style:
                      const TextStyle(color: Color(0xFFFCA5A5), fontSize: 12.5)),
            ],
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: _busy ? null : () => Navigator.pop(context),
                    child: const Text('Cancelar'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton(
                    onPressed: _busy ? null : _save,
                    style: FilledButton.styleFrom(
                      backgroundColor: AppColors.accent,
                      foregroundColor: Colors.black,
                    ),
                    child: _busy
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.black))
                        : Text(_isEdit ? 'Guardar' : 'Crear'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _field(TextEditingController c, String label, IconData icon) {
    return TextField(
      controller: c,
      style: TextStyle(color: AppColors.text),
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon, size: 20),
        border: const OutlineInputBorder(),
      ),
    );
  }
}
