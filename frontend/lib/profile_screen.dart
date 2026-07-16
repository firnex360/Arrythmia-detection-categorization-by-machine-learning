import 'package:flutter/material.dart';

import 'api_service.dart';
import 'models.dart';
import 'session.dart';
import 'theme.dart';

const _avatarColors = [
  '#38bdf8', '#10b981', '#f59e0b', '#ef4444',
  '#a855f7', '#ec4899', '#14b8a6', '#6366f1',
];

/// Lets the doctor edit their own profile: display name, avatar colour and
/// (optionally) their password.
class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  late final TextEditingController _name;
  late final TextEditingController _password;
  String? _avatarColor;
  bool _busy = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    final d = Session.doctor;
    _name = TextEditingController(text: d?.name ?? '');
    _password = TextEditingController();
    _avatarColor = d?.avatarColor ?? _avatarColors.first;
  }

  @override
  void dispose() {
    _name.dispose();
    _password.dispose();
    super.dispose();
  }

  Color get _color => PredictionResult.parseColor(_avatarColor, AppColors.accent);

  String get _initials {
    final parts =
        _name.text.trim().split(RegExp(r'\s+')).where((s) => s.isNotEmpty);
    if (parts.isEmpty) return '?';
    return parts.take(2).map((s) => s[0].toUpperCase()).join();
  }

  Future<void> _save() async {
    if (_name.text.trim().isEmpty) {
      setState(() => _error = 'El nombre es obligatorio.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await ApiService.updateProfile(
        name: _name.text.trim(),
        avatarColor: _avatarColor,
        password: _password.text.isEmpty ? null : _password.text,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('Perfil actualizado.')));
      Navigator.pop(context);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = '$e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final d = Session.doctor;
    return Scaffold(
      appBar: AppBar(title: const Text('Mi perfil')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 460),
          child: ListView(
            padding: const EdgeInsets.all(24),
            children: [
              Center(
                child: CircleAvatar(
                  radius: 44,
                  backgroundColor: _color.withValues(alpha: 0.2),
                  child: Text(_initials,
                      style: TextStyle(
                          color: _color,
                          fontSize: 30,
                          fontWeight: FontWeight.w800)),
                ),
              ),
              const SizedBox(height: 10),
              Center(
                child: Text('@${d?.username ?? ''}',
                    style: TextStyle(color: AppColors.muted, fontSize: 12.5)),
              ),
              const SizedBox(height: 24),
              TextField(
                controller: _name,
                onChanged: (_) => setState(() {}),
                style: TextStyle(color: AppColors.text),
                decoration: const InputDecoration(
                  labelText: 'Nombre para mostrar',
                  prefixIcon: Icon(Icons.badge_outlined, size: 20),
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 18),
              Text('Color del avatar',
                  style: TextStyle(
                      color: AppColors.text,
                      fontWeight: FontWeight.w600,
                      fontSize: 13)),
              const SizedBox(height: 10),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: [
                  for (final hex in _avatarColors)
                    GestureDetector(
                      onTap: () => setState(() => _avatarColor = hex),
                      child: Container(
                        width: 34,
                        height: 34,
                        decoration: BoxDecoration(
                          color: PredictionResult.parseColor(hex),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: _avatarColor == hex
                                ? AppColors.text
                                : Colors.transparent,
                            width: 2.5,
                          ),
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 20),
              TextField(
                controller: _password,
                obscureText: true,
                style: TextStyle(color: AppColors.text),
                decoration: const InputDecoration(
                  labelText: 'Nueva contraseña (opcional)',
                  helperText: 'Déjalo vacío para no cambiarla',
                  prefixIcon: Icon(Icons.lock_outline, size: 20),
                  border: OutlineInputBorder(),
                ),
              ),
              if (_error != null) ...[
                const SizedBox(height: 14),
                Text(_error!,
                    style: const TextStyle(
                        color: Color(0xFFFCA5A5), fontSize: 12.5)),
              ],
              const SizedBox(height: 22),
              FilledButton(
                onPressed: _busy ? null : _save,
                style: FilledButton.styleFrom(
                  minimumSize: const Size.fromHeight(48),
                  backgroundColor: AppColors.accent,
                  foregroundColor: Colors.black,
                ),
                child: _busy
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.black))
                    : const Text('Guardar cambios'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
