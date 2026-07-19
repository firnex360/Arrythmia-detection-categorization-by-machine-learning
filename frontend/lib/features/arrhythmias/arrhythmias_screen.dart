import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'package:frontend/core/theme.dart';

/// Reference info for one arrhythmia the model works with.
class ArrhythmiaInfo {
  final String code;
  final String name;
  final String summary;
  final List<String> features;
  final EcgPattern pattern;
  const ArrhythmiaInfo({
    required this.code,
    required this.name,
    required this.summary,
    required this.features,
    required this.pattern,
  });

  Color get color => arrhythmiaColor(code);
}

/// Shape parameters for the example waveform.
class EcgPattern {
  final double spacing; // horizontal distance between beats (px)
  final bool pWave; // visible P wave before QRS
  final bool irregular; // irregular R-R spacing
  final bool wavyBaseline; // fibrillatory baseline (no clear P)
  const EcgPattern({
    required this.spacing,
    this.pWave = true,
    this.irregular = false,
    this.wavyBaseline = false,
  });
}

const _arrhythmias = <ArrhythmiaInfo>[
  ArrhythmiaInfo(
    code: 'SR',
    name: 'Ritmo Sinusal (normal)',
    summary:
        'El corazón late de forma regular a 60–100 lpm con impulsos del nodo SA. '
        'Es el ritmo normal; no se detecta arritmia significativa.',
    features: [
      'Intervalos R-R regulares (espaciado uniforme entre latidos)',
      'Onda P clara y positiva (negativo en derivación aVR)',
      'PR constante entre 120–200 ms',
      'Frecuencia entre 60 y 100 lpm',
    ],
    pattern: EcgPattern(spacing: 90),
  ),
  ArrhythmiaInfo(
    code: 'AFIB',
    name: 'Fibrilación Auricular',
    summary:
        'Actividad eléctrica caótica en las aurículas que produce un ritmo '
        'irregular, con frecuencia rápido. Aumenta el riesgo de ictus.',
    features: [
      'Intervalos R-R irregularmente irregulares (sin patrón)',
      'Ausencia de ondas P — línea de base ondulante (ondas f)',
      'Respuesta ventricular a menudo rápida',
      'Complejos QRS de morfología conservada',
    ],
    pattern:
        EcgPattern(spacing: 70, pWave: false, irregular: true, wavyBaseline: true),
  ),
  ArrhythmiaInfo(
    code: 'STACH',
    name: 'Taquicardia Sinusal',
    summary:
        'Frecuencia cardíaca >100 lpm con ritmo regular originado en el nodo SA. '
        'Suele deberse a ejercicio, fiebre o estrés.',
    features: [
      'Frecuencia rápida (>100 lpm), latidos muy juntos',
      'Intervalos R-R regulares pese a la velocidad',
      'Onda P normal precediendo cada QRS',
      'Segmento T-P acortado (menos reposo entre latidos)',
    ],
    pattern: EcgPattern(spacing: 55),
  ),
  ArrhythmiaInfo(
    code: 'SBRAD',
    name: 'Bradicardia Sinusal',
    summary:
        'Frecuencia cardíaca <60 lpm con ritmo sinusal regular. Normal en '
        'atletas; en otros puede indicar problemas de conducción.',
    features: [
      'Frecuencia lenta (<60 lpm), latidos muy separados',
      'Intervalos R-R regulares',
      'Onda P normal y positiva antes de cada QRS',
      'Segmento T-P largo (reposo prolongado entre latidos)',
    ],
    pattern: EcgPattern(spacing: 135),
  ),
];

/// Quick reference for the arrhythmias the model can detect, each with an
/// illustrative example waveform.
class ArrhythmiasScreen extends StatefulWidget {
  const ArrhythmiasScreen({super.key});

  @override
  State<ArrhythmiasScreen> createState() => _ArrhythmiasScreenState();
}

class _ArrhythmiasScreenState extends State<ArrhythmiasScreen>
    with ThemeReactive<ArrhythmiasScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 900),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Arritmias del modelo',
                      style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          color: AppColors.text)),
                  const SizedBox(height: 2),
                  Text(
                    'Los cuatro ritmos que el modelo puede clasificar, con un '
                    'ejemplo de cómo se ve cada uno en el ECG.',
                    style: TextStyle(color: AppColors.muted, fontSize: 13),
                  ),
                  const SizedBox(height: 18),
                  for (final a in _arrhythmias) _ArrhythmiaCard(info: a),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ArrhythmiaCard extends StatelessWidget {
  final ArrhythmiaInfo info;
  const _ArrhythmiaCard({required this.info});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: info.color.withValues(alpha: 0.16),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(info.code,
                      style: TextStyle(
                          color: info.color,
                          fontWeight: FontWeight.w800,
                          fontSize: 14)),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(info.name,
                      style: TextStyle(
                          color: AppColors.text,
                          fontWeight: FontWeight.w700,
                          fontSize: 17)),
                ),
              ],
            ),
            const SizedBox(height: 14),
            // Example waveform illustration.
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Container(
                height: 120,
                width: double.infinity,
                color: AppColors.isDark
                    ? const Color(0xFF0A0E17)
                    : const Color(0xFFF8FAFC),
                child: CustomPaint(
                  painter: _EcgPainter(pattern: info.pattern, color: info.color),
                ),
              ),
            ),
            const SizedBox(height: 14),
            Text(info.summary,
                style: TextStyle(
                    color: AppColors.text, height: 1.5, fontSize: 13.5)),
            const SizedBox(height: 12),
            Text('Características clave',
                style: TextStyle(
                    color: AppColors.muted,
                    fontWeight: FontWeight.w600,
                    fontSize: 12)),
            const SizedBox(height: 6),
            for (final f in info.features)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(top: 3, right: 10),
                      child: Icon(Icons.circle, size: 7, color: info.color),
                    ),
                    Expanded(
                      child: Text(f,
                          style: TextStyle(
                              color: AppColors.text,
                              height: 1.4,
                              fontSize: 13)),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// Draws a representative ECG strip for a rhythm pattern.
class _EcgPainter extends CustomPainter {
  final EcgPattern pattern;
  final Color color;
  _EcgPainter({required this.pattern, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final baseline = size.height * 0.62;
    final amp = size.height * 0.30; // R-wave amplitude
    final line = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;

    final rand = math.Random(pattern.code.hashCode);
    final path = Path()..moveTo(0, baseline);

    // Relative beat template as (dx fraction of spacing, y fraction of amp).
    // y positive = upward (drawn as baseline - y*amp).
    List<List<double>> beat(double spacing) {
      final s = spacing;
      final pts = <List<double>>[];
      if (pattern.pWave) {
        pts.addAll([
          [0.12 * s, 0.0],
          [0.18 * s, 0.22],
          [0.24 * s, 0.0],
        ]);
      }
      pts.addAll([
        [0.40 * s, 0.0], // PR baseline
        [0.44 * s, -0.15], // Q
        [0.50 * s, 1.0], // R
        [0.56 * s, -0.30], // S
        [0.60 * s, 0.0], // back to baseline
        [0.74 * s, 0.0], // ST
        [0.80 * s, 0.32], // T
        [0.88 * s, 0.0],
      ]);
      return pts;
    }

    double x = 8;
    double y(double frac) => baseline - frac * amp;

    while (x < size.width - 8) {
      final spacing = pattern.irregular
          ? pattern.spacing * (0.55 + rand.nextDouble() * 0.9)
          : pattern.spacing;

      if (pattern.wavyBaseline) {
        // Fibrillatory baseline up to the QRS.
        for (double t = 0; t < 0.40 * spacing; t += 4) {
          final noise = (rand.nextDouble() - 0.5) * 0.14;
          path.lineTo(x + t, y(noise));
        }
      }

      for (final p in beat(spacing)) {
        path.lineTo(x + p[0], y(p[1]));
      }
      x += spacing;
    }

    canvas.drawPath(path, line);
  }

  @override
  bool shouldRepaint(covariant _EcgPainter old) =>
      old.pattern != pattern || old.color != color;
}

extension on EcgPattern {
  // Stable seed source so irregular/wavy strips look consistent per rhythm.
  String get code => '$spacing-$pWave-$irregular-$wavyBaseline';
}
