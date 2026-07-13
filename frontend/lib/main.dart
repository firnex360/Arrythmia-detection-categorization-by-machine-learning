import 'package:flutter/material.dart';

import 'login_screen.dart';
import 'theme.dart';

void main() {
  runApp(const EcgApp());
}

class EcgApp extends StatelessWidget {
  const EcgApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ECG Arrhythmia Detector',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: const LoginScreen(),
    );
  }
}
