import 'package:flutter/material.dart';

import 'package:frontend/features/auth/login_screen.dart';
import 'package:frontend/core/theme.dart';

void main() {
  runApp(const EcgApp());
}

class EcgApp extends StatelessWidget {
  const EcgApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Rebuild the whole app when the light/dark mode toggles.
    return ListenableBuilder(
      listenable: ThemeController.instance,
      builder: (context, _) {
        return MaterialApp(
          title: 'ECG IA',
          debugShowCheckedModeBanner: false,
          theme: buildAppTheme(),
          home: const LoginScreen(),
        );
      },
    );
  }
}
