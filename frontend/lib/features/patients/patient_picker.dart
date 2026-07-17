import 'package:flutter/material.dart';

import 'package:frontend/core/theme.dart';
import 'package:frontend/models/models.dart';
import 'package:frontend/services/api_service.dart';
import 'package:frontend/features/patients/patient_form.dart';

/// Bottom sheet to choose an existing patient or create a new one. Returns the
/// chosen [Patient], or null if cancelled.
Future<Patient?> showPatientPicker(BuildContext context) {
  return showModalBottomSheet<Patient>(
    context: context,
    isScrollControlled: true,
    backgroundColor: AppColors.surface,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
    ),
    builder: (_) => const _PatientPicker(),
  );
}

class _PatientPicker extends StatefulWidget {
  const _PatientPicker();

  @override
  State<_PatientPicker> createState() => _PatientPickerState();
}

class _PatientPickerState extends State<_PatientPicker>
    with ThemeReactive<_PatientPicker> {
  List<Patient>? _patients;
  String _query = '';
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final list = await ApiService.listPatients();
      if (mounted) setState(() => _patients = list);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  Future<void> _create() async {
    final created = await showPatientForm(context);
    if (created != null && mounted) Navigator.pop(context, created);
  }

  @override
  Widget build(BuildContext context) {
    final all = _patients;
    final items = (all ?? const <Patient>[]).where((p) {
      if (_query.trim().isEmpty) return true;
      final q = _query.toLowerCase();
      return p.name.toLowerCase().contains(q) ||
          (p.cedula ?? '').toLowerCase().contains(q);
    }).toList();

    return SafeArea(
      child: Padding(
        padding: EdgeInsets.only(
            bottom: MediaQuery.of(context).viewInsets.bottom),
        child: SizedBox(
          height: MediaQuery.of(context).size.height * 0.7,
          child: Column(
            children: [
              const SizedBox(height: 12),
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: AppColors.border,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 8),
                child: Row(
                  children: [
                    Text('Asignar a un paciente',
                        style: TextStyle(
                            color: AppColors.text,
                            fontWeight: FontWeight.w800,
                            fontSize: 17)),
                    const Spacer(),
                    FilledButton.icon(
                      onPressed: _create,
                      style: FilledButton.styleFrom(
                          backgroundColor: AppColors.accent,
                          foregroundColor: Colors.black),
                      icon: const Icon(Icons.person_add_alt_1, size: 18),
                      label: const Text('Nuevo'),
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: TextField(
                  onChanged: (v) => setState(() => _query = v),
                  style: TextStyle(color: AppColors.text),
                  decoration: InputDecoration(
                    hintText: 'Buscar por nombre o cédula…',
                    prefixIcon: const Icon(Icons.search, size: 20),
                    filled: true,
                    fillColor: AppColors.surface2,
                    border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide(color: AppColors.border)),
                    enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide(color: AppColors.border)),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Expanded(
                child: _error != null
                    ? Center(
                        child: Text(_error!,
                            style: TextStyle(color: AppColors.muted)))
                    : all == null
                        ? const Center(child: CircularProgressIndicator())
                        : items.isEmpty
                            ? Center(
                                child: Text(
                                  all.isEmpty
                                      ? 'No tienes pacientes. Crea uno con “Nuevo”.'
                                      : 'Ningún paciente coincide.',
                                  style: TextStyle(color: AppColors.muted),
                                ),
                              )
                            : ListView.builder(
                                padding:
                                    const EdgeInsets.fromLTRB(16, 4, 16, 16),
                                itemCount: items.length,
                                itemBuilder: (_, i) {
                                  final p = items[i];
                                  return Card(
                                    margin: const EdgeInsets.only(bottom: 8),
                                    child: ListTile(
                                      leading: CircleAvatar(
                                        backgroundColor: AppColors.accent
                                            .withValues(alpha: 0.15),
                                        child: Text(p.initials,
                                            style: TextStyle(
                                                color: AppColors.accent,
                                                fontWeight: FontWeight.w700)),
                                      ),
                                      title: Text(p.name,
                                          style: TextStyle(
                                              color: AppColors.text,
                                              fontWeight: FontWeight.w600)),
                                      subtitle: Text(
                                        [
                                          if (p.cedula != null &&
                                              p.cedula!.isNotEmpty)
                                            'CC ${p.cedula}',
                                          if (p.age != null) '${p.age} años',
                                          p.genderLabel,
                                        ].join(' · '),
                                        style: TextStyle(
                                            color: AppColors.muted,
                                            fontSize: 12),
                                      ),
                                      onTap: () => Navigator.pop(context, p),
                                    ),
                                  );
                                },
                              ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
