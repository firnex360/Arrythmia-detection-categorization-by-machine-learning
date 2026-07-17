import 'package:flutter/material.dart';

import 'package:frontend/services/api_service.dart';
import 'package:frontend/features/shell/app_shell.dart';
import 'package:frontend/core/config.dart';
import 'package:frontend/core/theme.dart';

/// Doctor sign-in / sign-up. On success, replaces itself with the patients list.
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> with ThemeReactive<LoginScreen> {
  final _username = TextEditingController(text: 'admin');
  final _password = TextEditingController(text: 'admin');
  final _name = TextEditingController();

  bool _registerMode = false;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _username.dispose();
    _password.dispose();
    _name.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      if (_registerMode) {
        await ApiService.register(
            _username.text.trim(), _password.text, _name.text.trim());
      } else {
        await ApiService.login(_username.text.trim(), _password.text);
      }
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const AppShell()),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _editServerUrl() async {
    final controller = TextEditingController(text: AppConfig.baseUrl);
    final url = await showDialog<String>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: Text('Servidor backend'),
        content: TextField(
          controller: controller,
          decoration: InputDecoration(
            border: OutlineInputBorder(),
            hintText: 'http://127.0.0.1:5000',
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text('Cancelar')),
          FilledButton(
              onPressed: () => Navigator.pop(context, controller.text.trim()),
              child: Text('Guardar')),
        ],
      ),
    );
    if (url != null && url.isNotEmpty) {
      setState(() => AppConfig.baseUrl = url);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        actions: [
          IconButton(
            tooltip: 'Servidor',
            onPressed: _busy ? null : _editServerUrl,
            icon: Icon(Icons.dns_outlined),
          ),
        ],
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 420),
          child: ListView(
            padding: EdgeInsets.all(24),
            shrinkWrap: true,
            children: [
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  shape: BoxShape.circle,
                  border: Border.all(color: AppColors.border),
                ),
                child: Icon(Icons.monitor_heart_rounded,
                    size: 44, color: AppColors.accent),
              ).let((w) => Center(child: w)),
              SizedBox(height: 18),
              Text(
                _registerMode ? 'Crear cuenta de doctor' : '¡Bienvenido, Doctor!',
                textAlign: TextAlign.center,
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    color: AppColors.text),
              ),
              SizedBox(height: 6),
              Text(
                _registerMode
                    ? 'Registra tus credenciales para acceder a tus pacientes.'
                    : 'Ingresa para ver el historial de tus pacientes.',
                textAlign: TextAlign.center,
                style: TextStyle(color: AppColors.muted, fontSize: 13),
              ),
              SizedBox(height: 24),
              if (_registerMode) ...[
                _field(_name, 'Nombre completo', Icons.badge_outlined),
                SizedBox(height: 12),
              ],
              _field(_username, 'Usuario', Icons.person_outline),
              SizedBox(height: 12),
              _field(_password, 'Contraseña', Icons.lock_outline,
                  obscure: true, onSubmit: (_) => _submit()),
              if (_error != null) ...[
                SizedBox(height: 14),
                Container(
                  padding: EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color(0x22EF4444),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: const Color(0x55EF4444)),
                  ),
                  child: Text(_error!,
                      style: TextStyle(
                          color: Color(0xFFFCA5A5), fontSize: 12.5)),
                ),
              ],
              SizedBox(height: 20),
              FilledButton(
                onPressed: _busy ? null : _submit,
                style: FilledButton.styleFrom(
                  minimumSize: const Size.fromHeight(48),
                  backgroundColor: AppColors.accent,
                  foregroundColor: Colors.black,
                ),
                child: _busy
                    ? SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.black))
                    : Text(_registerMode ? 'Registrarse' : 'Ingresar'),
              ),
              SizedBox(height: 8),
              TextButton(
                onPressed: _busy
                    ? null
                    : () => setState(() {
                          _registerMode = !_registerMode;
                          _error = null;
                        }),
                child: Text(_registerMode
                    ? '¿Ya tienes cuenta? Ingresar'
                    : '¿Nuevo? Crear una cuenta'),
              ),
              if (!_registerMode)
                Text(
                  'Cuenta demo: admin / admin',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: AppColors.muted, fontSize: 11.5),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _field(
    TextEditingController c,
    String label,
    IconData icon, {
    bool obscure = false,
    ValueChanged<String>? onSubmit,
  }) {
    return TextField(
      controller: c,
      obscureText: obscure,
      onSubmitted: onSubmit,
      style: TextStyle(color: AppColors.text),
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon, size: 20),
        border: OutlineInputBorder(),
      ),
    );
  }
}

extension<T> on T {
  R let<R>(R Function(T) f) => f(this);
}
