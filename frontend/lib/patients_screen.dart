import 'package:flutter/material.dart';

import 'api_service.dart';
import 'dashboard_screen.dart';
import 'login_screen.dart';
import 'risk_screen.dart';
import 'models.dart';
import 'patient_detail_screen.dart';
import 'patient_form.dart';
import 'session.dart';
import 'theme.dart';

/// Home screen after login: the doctor's own patients, searchable, with entry
/// points to add a patient and to the global dashboard.
class PatientsScreen extends StatefulWidget {
  const PatientsScreen({super.key});

  @override
  State<PatientsScreen> createState() => _PatientsScreenState();
}

class _PatientsScreenState extends State<PatientsScreen> {
  List<Patient>? _patients;
  String? _error;
  String _query = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final list = await ApiService.listPatients();
      if (!mounted) return;
      setState(() => _patients = list);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _addPatient() async {
    final created = await showPatientForm(context);
    if (created != null) _load();
  }

  Future<void> _logout() async {
    await ApiService.logout();
    if (!mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const LoginScreen()),
      (_) => false,
    );
  }

  List<Patient> get _filtered {
    final all = _patients ?? const <Patient>[];
    if (_query.trim().isEmpty) return all;
    final q = _query.toLowerCase();
    return all.where((p) => p.name.toLowerCase().contains(q)).toList();
  }

  @override
  Widget build(BuildContext context) {
    final doctorName = Session.doctor?.name ?? 'Doctor';
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mis pacientes'),
        actions: [
          IconButton(
            tooltip: 'Riesgo y alertas',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const RiskScreen()),
            ),
            icon: const Icon(Icons.warning_amber_rounded),
          ),
          IconButton(
            tooltip: 'Dashboard poblacional',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const DashboardScreen()),
            ),
            icon: const Icon(Icons.insights_rounded),
          ),
          IconButton(
            tooltip: 'Cerrar sesión',
            onPressed: _logout,
            icon: const Icon(Icons.logout_rounded),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _addPatient,
        backgroundColor: AppColors.accent,
        foregroundColor: Colors.black,
        icon: const Icon(Icons.person_add_alt_1),
        label: const Text('Nuevo paciente'),
      ),
      body: RefreshIndicator(
        onRefresh: _load,
        child: _buildBody(doctorName),
      ),
    );
  }

  Widget _buildBody(String doctorName) {
    if (_error != null) {
      return _ErrorView(message: _error!, onRetry: _load);
    }
    if (_patients == null) {
      return const Center(child: CircularProgressIndicator());
    }

    final items = _filtered;
    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        // Card grid on wide screens; single column on phones.
        final cols = w >= 1100 ? 3 : (w >= 720 ? 2 : 1);
        const gap = 14.0;

        Widget tiles;
        if (items.isEmpty) {
          tiles = _EmptyView(hasPatients: _patients!.isNotEmpty);
        } else if (cols == 1) {
          tiles = Column(
            children: [
              for (final p in items)
                _PatientTile(patient: p, onChanged: _load),
            ],
          );
        } else {
          final contentW = w.clamp(0, 1160).toDouble() -
              (w >= 900 ? 64 : 32); // account for padding
          final tileW = (contentW - gap * (cols - 1)) / cols;
          tiles = Wrap(
            spacing: gap,
            runSpacing: gap,
            children: [
              for (final p in items)
                SizedBox(
                    width: tileW,
                    child: _PatientTile(patient: p, onChanged: _load)),
            ],
          );
        }

        return ListView(
          padding: EdgeInsets.symmetric(
              horizontal: w >= 900 ? 32 : 16, vertical: 12),
          children: [
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1160),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Hola, $doctorName',
                        style: const TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                            color: AppColors.text)),
                    const SizedBox(height: 2),
                    Text('${_patients!.length} paciente(s) registrados',
                        style: const TextStyle(
                            color: AppColors.muted, fontSize: 12.5)),
                    const SizedBox(height: 16),
                    ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 480),
                      child: TextField(
                        onChanged: (v) => setState(() => _query = v),
                        style: const TextStyle(color: AppColors.text),
                        decoration: InputDecoration(
                          hintText: 'Buscar por nombre…',
                          prefixIcon: const Icon(Icons.search, size: 20),
                          filled: true,
                          fillColor: AppColors.surface,
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12),
                            borderSide: const BorderSide(color: AppColors.border),
                          ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12),
                            borderSide: const BorderSide(color: AppColors.border),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 18),
                    tiles,
                    const SizedBox(height: 80),
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

class _PatientTile extends StatelessWidget {
  final Patient patient;
  final VoidCallback onChanged;
  const _PatientTile({required this.patient, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    final initials = patient.name.isNotEmpty
        ? patient.name.trim().split(RegExp(r'\s+')).take(2).map((s) => s[0]).join()
        : '?';
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
        leading: CircleAvatar(
          backgroundColor: const Color(0x2638BDF8),
          child: Text(initials.toUpperCase(),
              style: const TextStyle(
                  color: AppColors.accent, fontWeight: FontWeight.w700)),
        ),
        title: Text(patient.name,
            style: const TextStyle(
                color: AppColors.text, fontWeight: FontWeight.w600)),
        subtitle: Text(
          [
            if (patient.age != null) '${patient.age} años',
            patient.genderLabel,
            '${patient.recordCount} ECG',
          ].join(' · '),
          style: const TextStyle(color: AppColors.muted, fontSize: 12),
        ),
        trailing: const Icon(Icons.chevron_right, color: AppColors.muted),
        onTap: () async {
          await Navigator.of(context).push(
            MaterialPageRoute(
                builder: (_) => PatientDetailScreen(patientId: patient.id)),
          );
          onChanged(); // refresh counts after returning
        },
      ),
    );
  }
}

class _EmptyView extends StatelessWidget {
  final bool hasPatients;
  const _EmptyView({required this.hasPatients});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 60),
      child: Column(
        children: [
          Icon(hasPatients ? Icons.search_off : Icons.people_outline,
              size: 54, color: AppColors.muted),
          const SizedBox(height: 12),
          Text(
            hasPatients
                ? 'Ningún paciente coincide con la búsqueda.'
                : 'Aún no tienes pacientes.\nAgrega el primero con el botón “Nuevo paciente”.',
            textAlign: TextAlign.center,
            style: const TextStyle(color: AppColors.muted, height: 1.5),
          ),
        ],
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  const _ErrorView({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return ListView(
      children: [
        const SizedBox(height: 80),
        const Icon(Icons.cloud_off_rounded, size: 54, color: AppColors.muted),
        const SizedBox(height: 12),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Text(message,
              textAlign: TextAlign.center,
              style: const TextStyle(color: AppColors.muted, height: 1.5)),
        ),
        const SizedBox(height: 16),
        Center(
          child: OutlinedButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh),
            label: const Text('Reintentar'),
          ),
        ),
      ],
    );
  }
}
