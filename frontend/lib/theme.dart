import 'package:flutter/material.dart';

/// Dark clinical palette, matching the existing web UI (templates/index.html).
class AppColors {
  static const bg = Color(0xFF0B0F1A);
  static const surface = Color(0xFF111827);
  static const surface2 = Color(0xFF1A2235);
  static const border = Color(0x14FFFFFF);
  static const text = Color(0xFFE2E8F0);
  static const muted = Color(0xFF64748B);
  static const accent = Color(0xFF38BDF8);
}

ThemeData buildAppTheme() {
  final base = ThemeData.dark(useMaterial3: true);
  return base.copyWith(
    scaffoldBackgroundColor: AppColors.bg,
    colorScheme: base.colorScheme.copyWith(
      primary: AppColors.accent,
      surface: AppColors.surface,
    ),
    cardTheme: const CardThemeData(
      color: AppColors.surface,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(14)),
        side: BorderSide(color: AppColors.border),
      ),
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.bg,
      elevation: 0,
      centerTitle: true,
    ),
    textTheme: base.textTheme.apply(
      bodyColor: AppColors.text,
      displayColor: AppColors.text,
    ),
  );
}
