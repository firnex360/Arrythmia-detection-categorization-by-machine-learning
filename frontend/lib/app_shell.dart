import 'package:flutter/material.dart';

import 'analysis_screen.dart';
import 'api_service.dart';
import 'dashboard_screen.dart';
import 'login_screen.dart';
import 'patients_screen.dart';
import 'profile_screen.dart';
import 'risk_screen.dart';
import 'session.dart';
import 'theme.dart';

class _NavItem {
  final IconData icon;
  final String label;
  const _NavItem(this.icon, this.label);
}

const _navItems = [
  _NavItem(Icons.upload_file_rounded, 'Análisis'),
  _NavItem(Icons.people_alt_rounded, 'Pacientes'),
  _NavItem(Icons.insights_rounded, 'Dashboard'),
  _NavItem(Icons.warning_amber_rounded, 'Riesgo'),
];

/// Persistent app frame: left navigation sidebar + top bar with profile menu and
/// light/dark toggle. Hosts the four main sections.
class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _index = 0;

  Widget _bodyFor(int i) => switch (i) {
        0 => const AnalysisScreen(),
        1 => const PatientsScreen(),
        2 => const DashboardScreen(),
        _ => const RiskScreen(),
      };

  Future<void> _openProfile() async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const ProfileScreen()),
    );
    if (mounted) setState(() {}); // name/avatar may have changed
  }

  Future<void> _logout() async {
    await ApiService.logout();
    if (!mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const LoginScreen()),
      (_) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    final w = MediaQuery.of(context).size.width;
    final compact = w < 1040; // icon-only rail on smaller screens

    return Scaffold(
      body: SafeArea(
        child: Row(
          children: [
            _Sidebar(
              index: _index,
              compact: compact,
              onSelect: (i) => setState(() => _index = i),
            ),
            Container(width: 1, color: AppColors.border),
            Expanded(
              child: Column(
                children: [
                  _TopBar(
                    title: _navItems[_index].label,
                    onProfile: _openProfile,
                    onLogout: _logout,
                  ),
                  Container(height: 1, color: AppColors.border),
                  Expanded(child: _bodyFor(_index)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Sidebar extends StatelessWidget {
  final int index;
  final bool compact;
  final ValueChanged<int> onSelect;

  const _Sidebar({
    required this.index,
    required this.compact,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: compact ? 76 : 216,
      color: AppColors.surface,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: EdgeInsets.symmetric(
                horizontal: compact ? 8 : 18, vertical: 22),
            child: Row(
              mainAxisAlignment:
                  compact ? MainAxisAlignment.center : MainAxisAlignment.start,
              children: [
                Icon(Icons.monitor_heart_rounded,
                    color: AppColors.accent, size: 26),
                if (!compact) ...[
                  const SizedBox(width: 10),
                  Text('ECG IA',
                      style: TextStyle(
                          color: AppColors.text,
                          fontSize: 17,
                          fontWeight: FontWeight.w800)),
                ],
              ],
            ),
          ),
          const SizedBox(height: 6),
          for (var i = 0; i < _navItems.length; i++)
            _NavTile(
              item: _navItems[i],
              selected: i == index,
              compact: compact,
              onTap: () => onSelect(i),
            ),
          const Spacer(),
          _ThemeToggleTile(compact: compact),
          const SizedBox(height: 10),
        ],
      ),
    );
  }
}

class _NavTile extends StatelessWidget {
  final _NavItem item;
  final bool selected;
  final bool compact;
  final VoidCallback onTap;

  const _NavTile({
    required this.item,
    required this.selected,
    required this.compact,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final fg = selected ? AppColors.accent : AppColors.muted;
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: compact ? 10 : 12, vertical: 3),
      child: Material(
        color: selected ? AppColors.accent.withValues(alpha: 0.14) : Colors.transparent,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onTap,
          child: Padding(
            padding: EdgeInsets.symmetric(
                horizontal: compact ? 0 : 14, vertical: 12),
            child: Row(
              mainAxisAlignment: compact
                  ? MainAxisAlignment.center
                  : MainAxisAlignment.start,
              children: [
                Icon(item.icon, color: fg, size: 22),
                if (!compact) ...[
                  const SizedBox(width: 14),
                  Text(item.label,
                      style: TextStyle(
                          color: selected ? AppColors.text : AppColors.muted,
                          fontWeight:
                              selected ? FontWeight.w700 : FontWeight.w500,
                          fontSize: 14)),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _ThemeToggleTile extends StatelessWidget {
  final bool compact;
  const _ThemeToggleTile({required this.compact});

  @override
  Widget build(BuildContext context) {
    final dark = ThemeController.instance.isDark;
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: compact ? 10 : 12),
      child: Material(
        color: Colors.transparent,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () => ThemeController.instance.toggle(),
          child: Padding(
            padding: EdgeInsets.symmetric(
                horizontal: compact ? 0 : 14, vertical: 12),
            child: Row(
              mainAxisAlignment: compact
                  ? MainAxisAlignment.center
                  : MainAxisAlignment.start,
              children: [
                Icon(dark ? Icons.dark_mode_outlined : Icons.light_mode_outlined,
                    color: AppColors.muted, size: 20),
                if (!compact) ...[
                  const SizedBox(width: 14),
                  Text(dark ? 'Modo oscuro' : 'Modo claro',
                      style: TextStyle(color: AppColors.muted, fontSize: 13)),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _TopBar extends StatelessWidget {
  final String title;
  final VoidCallback onProfile;
  final VoidCallback onLogout;

  const _TopBar({
    required this.title,
    required this.onProfile,
    required this.onLogout,
  });

  @override
  Widget build(BuildContext context) {
    final doctor = Session.doctor;
    return Container(
      height: 62,
      color: AppColors.bg,
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: [
          Text(title,
              style: TextStyle(
                  color: AppColors.text,
                  fontSize: 18,
                  fontWeight: FontWeight.w700)),
          const Spacer(),
          PopupMenuButton<String>(
            tooltip: 'Perfil',
            color: AppColors.surface,
            onSelected: (v) {
              if (v == 'profile') onProfile();
              if (v == 'logout') onLogout();
            },
            itemBuilder: (_) => [
              PopupMenuItem(
                value: 'profile',
                child: Row(children: [
                  Icon(Icons.person_outline, size: 18, color: AppColors.text),
                  const SizedBox(width: 10),
                  const Text('Editar perfil'),
                ]),
              ),
              PopupMenuItem(
                value: 'logout',
                child: Row(children: [
                  Icon(Icons.logout_rounded, size: 18, color: AppColors.text),
                  const SizedBox(width: 10),
                  const Text('Cerrar sesión'),
                ]),
              ),
            ],
            child: Row(
              children: [
                if (doctor != null) ...[
                  Text(doctor.name,
                      style: TextStyle(
                          color: AppColors.text,
                          fontWeight: FontWeight.w600,
                          fontSize: 13.5)),
                  const SizedBox(width: 10),
                ],
                CircleAvatar(
                  radius: 17,
                  backgroundColor: (doctor?.color ?? AppColors.accent)
                      .withValues(alpha: 0.22),
                  child: Text(
                    doctor?.initials ?? '?',
                    style: TextStyle(
                        color: doctor?.color ?? AppColors.accent,
                        fontWeight: FontWeight.w700,
                        fontSize: 13),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
