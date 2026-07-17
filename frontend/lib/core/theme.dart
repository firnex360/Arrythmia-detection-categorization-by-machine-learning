import 'package:flutter/material.dart';

/// App palette that swaps between dark and light. Members are getters (not
/// `const`) so the whole UI recolours when [ThemeController] toggles the mode.
class AppColors {
  AppColors._();

  /// Current mode. Kept in sync by [ThemeController].
  static bool isDark = true;

  // ── Dark palette ──
  static const _dBg = Color(0xFF0B0F1A);
  static const _dSurface = Color(0xFF111827);
  static const _dSurface2 = Color(0xFF1A2235);
  static const _dBorder = Color(0x14FFFFFF);
  static const _dText = Color(0xFFE2E8F0);
  static const _dMuted = Color(0xFF64748B);

  // ── Light palette ──
  static const _lBg = Color(0xFFF5F7FB);
  static const _lSurface = Color(0xFFFFFFFF);
  static const _lSurface2 = Color(0xFFEEF2F7);
  static const _lBorder = Color(0x14000000);
  static const _lText = Color(0xFF0F172A);
  static const _lMuted = Color(0xFF64748B);

  static Color get bg => isDark ? _dBg : _lBg;
  static Color get surface => isDark ? _dSurface : _lSurface;
  static Color get surface2 => isDark ? _dSurface2 : _lSurface2;
  static Color get border => isDark ? _dBorder : _lBorder;
  static Color get text => isDark ? _dText : _lText;
  static Color get muted => isDark ? _dMuted : _lMuted;
  static const Color accent = Color(0xFF38BDF8);
}

/// Global light/dark switch. Session-scoped (in memory) like the auth token.
class ThemeController extends ChangeNotifier {
  ThemeController._();
  static final ThemeController instance = ThemeController._();

  bool get isDark => AppColors.isDark;
  ThemeMode get mode => AppColors.isDark ? ThemeMode.dark : ThemeMode.light;

  void toggle() => set(!AppColors.isDark);

  void set(bool dark) {
    if (AppColors.isDark == dark) return;
    AppColors.isDark = dark;
    notifyListeners();
  }
}

/// Mixin for screen States so they rebuild when the light/dark mode toggles.
///
/// Needed because our colours are global getters (`AppColors.x`) rather than
/// `Theme.of(context)` reads, so widgets don't otherwise depend on the theme and
/// wouldn't repaint their text/borders on a toggle.
mixin ThemeReactive<T extends StatefulWidget> on State<T> {
  @override
  void initState() {
    super.initState();
    ThemeController.instance.addListener(_onThemeChanged);
  }

  @override
  void dispose() {
    ThemeController.instance.removeListener(_onThemeChanged);
    super.dispose();
  }

  void _onThemeChanged() {
    if (mounted) setState(() {});
  }
}

/// Per-class accent colours, matching the backend's `_COLORS` in app.py.
const Map<String, Color> kArrhythmiaColors = {
  'SR': Color(0xFF10B981),
  'AFIB': Color(0xFFEF4444),
  'STACH': Color(0xFFF59E0B),
  'SBRAD': Color(0xFF3B82F6),
};

Color arrhythmiaColor(String code) => kArrhythmiaColors[code] ?? AppColors.accent;

ThemeData buildAppTheme() {
  final dark = AppColors.isDark;
  final base = dark ? ThemeData.dark(useMaterial3: true)
                    : ThemeData.light(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: AppColors.bg,
    colorScheme: base.colorScheme.copyWith(
      primary: AppColors.accent,
      surface: AppColors.surface,
      brightness: dark ? Brightness.dark : Brightness.light,
    ),
    cardTheme: CardThemeData(
      color: AppColors.surface,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: const BorderRadius.all(Radius.circular(14)),
        side: BorderSide(color: AppColors.border),
      ),
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: AppColors.bg,
      foregroundColor: AppColors.text,
      elevation: 0,
      centerTitle: true,
    ),
    dividerColor: AppColors.border,
    textTheme: base.textTheme.apply(
      bodyColor: AppColors.text,
      displayColor: AppColors.text,
    ),
  );
}
