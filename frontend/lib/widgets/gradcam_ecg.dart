import 'package:flutter/material.dart';

import 'package:frontend/core/theme.dart';

/// Draws an ECG trace and colours it by Grad-CAM importance.
///
/// This is the "explainability" visual: the waveform is the actual signal the
/// model saw (one lead), and each segment is tinted from cool (ignored) to hot
/// (heavily weighted) using the per-time-step importance the backend computed.
/// Where the line glows red/orange is where the model focused when it reached
/// its verdict.
class GradCamEcg extends StatelessWidget {
  final List<double> signal; // raw ECG samples for one lead
  final List<double> importance; // Grad-CAM 0..1, same length as signal
  final String leadLabel;

  const GradCamEcg({
    super.key,
    required this.signal,
    required this.importance,
    required this.leadLabel,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text('Derivación $leadLabel',
                style: TextStyle(
                    color: AppColors.muted,
                    fontWeight: FontWeight.w600,
                    fontSize: 13)),
            SizedBox(width: 14),
            const _RrLegend(),
            Spacer(),
            const _HeatLegend(),
          ],
        ),
        SizedBox(height: 8),
        AspectRatio(
          aspectRatio: 2.4,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Container(
              // Fondo oscuro en modo oscuro; blanco (tipo papel de ECG) en claro.
              color: AppColors.isDark ? const Color(0xFF0A0E17) : Colors.white,
              child: CustomPaint(
                painter: _GradCamPainter(
                  signal: signal,
                  importance: importance,
                  isDark: AppColors.isDark,
                ),
                size: Size.infinite,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// Leyenda que aclara que los círculos del trazado marcan los picos R (R-R).
class _RrLegend extends StatelessWidget {
  const _RrLegend();

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 11,
          height: 11,
          decoration: BoxDecoration(
            // Mismo color que el marcador del trazado, para que la leyenda
            // coincida y siga siendo visible en modo claro.
            color: AppColors.isDark ? Colors.white : const Color(0xFF0F172A),
            shape: BoxShape.circle,
            border: Border.all(color: const Color(0xFFEF4444), width: 1.5),
          ),
        ),
        SizedBox(width: 6),
        Text('picos R-R',
            style: TextStyle(color: AppColors.muted, fontSize: 11)),
      ],
    );
  }
}

class _HeatLegend extends StatelessWidget {
  const _HeatLegend();

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('bajo',
            style: TextStyle(color: AppColors.muted, fontSize: 11)),
        SizedBox(width: 6),
        Container(
          width: 70,
          height: 8,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(4),
            gradient: LinearGradient(
              colors: [
                Color(0xFF1E3A8A),
                Color(0xFF10B981),
                Color(0xFFF59E0B),
                Color(0xFFEF4444),
              ],
            ),
          ),
        ),
        SizedBox(width: 6),
        Text('alto',
            style: TextStyle(color: AppColors.muted, fontSize: 11)),
      ],
    );
  }
}

class _GradCamPainter extends CustomPainter {
  final List<double> signal;
  final List<double> importance;

  /// Modo del tema: cambia la rejilla y los marcadores R-R para que contrasten
  /// tanto sobre el fondo oscuro como sobre el blanco del modo claro.
  final bool isDark;

  _GradCamPainter({
    required this.signal,
    required this.importance,
    required this.isDark,
  });

  // Colores de los marcadores R-R según el tema.
  Color get _rrDot => isDark ? Colors.white : const Color(0xFF0F172A);
  Color get _rrLine =>
      isDark ? const Color(0x99FFFFFF) : const Color(0x99334155);
  Color get _rrGuide =>
      isDark ? const Color(0x33FFFFFF) : const Color(0x22000000);

  /// Maps a 0..1 importance value to a cool→hot colour.
  static Color heatColor(double t) {
    t = t.clamp(0.0, 1.0);
    const stops = [
      Color(0xFF1E3A8A), // deep blue  — ignored
      Color(0xFF10B981), // green
      Color(0xFFF59E0B), // amber
      Color(0xFFEF4444), // red        — most important
    ];
    final scaled = t * (stops.length - 1);
    final i = scaled.floor().clamp(0, stops.length - 2);
    final f = scaled - i;
    return Color.lerp(stops[i], stops[i + 1], f)!;
  }

  @override
  void paint(Canvas canvas, Size size) {
    _drawEcgGrid(canvas, size);
    if (signal.isEmpty) return;

    // Vertical scaling based on the signal's own range, with a little padding.
    var minV = signal.first, maxV = signal.first;
    for (final v in signal) {
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    final range = (maxV - minV).abs() < 1e-6 ? 1.0 : (maxV - minV);
    final pad = size.height * 0.12;

    double xOf(int i) => size.width * i / (signal.length - 1);
    double yOf(double v) =>
        size.height - pad - (v - minV) / range * (size.height - 2 * pad);

    // Guías verticales de cada pico R (debajo del trazado) para que el ritmo
    // R-R se note de un vistazo.
    final peaks = _findRPeaks();
    _drawRrGuides(canvas, size, peaks, xOf);

    // Draw the trace as many short coloured segments so the colour can follow
    // the Grad-CAM importance along the time axis.
    final paint = Paint()
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    for (var i = 0; i < signal.length - 1; i++) {
      final imp = i < importance.length ? importance[i] : 0.0;
      paint.color = heatColor(imp);
      // Add a soft glow on the hottest segments.
      if (imp > 0.6) {
        paint.maskFilter = const MaskFilter.blur(BlurStyle.solid, 1.5);
      } else {
        paint.maskFilter = null;
      }
      canvas.drawLine(
        Offset(xOf(i), yOf(signal[i])),
        Offset(xOf(i + 1), yOf(signal[i + 1])),
        paint,
      );
    }
    paint.maskFilter = null;

    // ── Marcadores R-R ────────────────────────────────────────────────────
    // LOS CÍRCULOS RESALTAN LOS PUNTOS R-R DEL ELECTROCARDIOGRAMA: cada círculo
    // (punto blanco con anillo rojo) marca el pico de una onda R, y la línea que
    // los une representa el intervalo R-R, es decir, la distancia entre latidos
    // consecutivos — de ahí se lee la frecuencia y la regularidad del ritmo.
    _drawRrMarkers(canvas, peaks, xOf, yOf);
  }

  /// Detecta los picos R (las deflexiones altas del QRS) para poder resaltar el
  /// ritmo latido a latido.
  ///
  /// Se hace en dos pasos para que el marcador caiga siempre sobre el pico real:
  ///  1. Umbral robusto por percentiles (la mediana como línea isoeléctrica y el
  ///     percentil 99 como tope de las ondas R). Usar percentiles en vez de
  ///     mín/máx evita que un artefacto aislado o una onda S profunda desplacen
  ///     el umbral y se pierdan latidos.
  ///  2. Se agrupan las muestras que superan el umbral en regiones (cada región
  ///     es un QRS) y se toma la muestra MÁS ALTA de cada región. Así no se marca
  ///     el primer punto que cruza el umbral, que era el error anterior.
  List<int> _findRPeaks() {
    final n = signal.length;
    if (n < 3) return const [];

    final sorted = List<double>.from(signal)..sort();
    double pct(double p) => sorted[(p * (n - 1)).round().clamp(0, n - 1)];
    final baseline = pct(0.50); // mediana ≈ línea isoeléctrica
    final high = pct(0.99); // tope de las ondas R
    final amplitude = high - baseline;
    if (amplitude <= 1e-9) return const []; // señal plana

    final threshold = baseline + 0.45 * amplitude;

    // 1) Máximo de cada región por encima del umbral = pico R.
    final candidates = <int>[];
    var i = 0;
    while (i < n) {
      if (signal[i] > threshold) {
        var best = i;
        while (i < n && signal[i] > threshold) {
          if (signal[i] > signal[best]) best = i;
          i++;
        }
        candidates.add(best);
      } else {
        i++;
      }
    }

    // 2) Periodo refractario: si dos picos quedan demasiado cerca para ser
    //    latidos distintos, nos quedamos con el MÁS ALTO (antes se conservaba el
    //    primero, que podía ser una muesca del QRS y no la onda R).
    //
    //    36 muestras @100 Hz = 0.36 s, la ventana clásica de rechazo de onda T:
    //    la onda T aparece ~0.30 s después de la R y, al ser más baja, se
    //    descarta aquí. Con 0.30 s se colaba y en bradicardia llegaba a duplicar
    //    los latidos detectados. El techo queda en ~166 lpm, de sobra para las
    //    clases del modelo.
    const minDist = 36;
    final peaks = <int>[];
    for (final c in candidates) {
      if (peaks.isNotEmpty && c - peaks.last < minDist) {
        if (signal[c] > signal[peaks.last]) peaks[peaks.length - 1] = c;
      } else {
        peaks.add(c);
      }
    }
    return peaks;
  }

  /// Guías verticales tenues en cada pico R — marcan visualmente la separación
  /// R-R a lo largo de todo el trazado.
  void _drawRrGuides(
      Canvas canvas, Size size, List<int> peaks, double Function(int) xOf) {
    final guide = Paint()
      ..color = _rrGuide
      ..strokeWidth = 1;
    for (final i in peaks) {
      final x = xOf(i);
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), guide);
    }
  }

  /// Dibuja los marcadores R-R sobre el trazado: la línea que conecta picos R
  /// consecutivos (el intervalo R-R) y un círculo en cada pico R.
  void _drawRrMarkers(Canvas canvas, List<int> peaks, double Function(int) xOf,
      double Function(double) yOf) {
    if (peaks.isEmpty) return;

    final points = [
      for (final i in peaks) Offset(xOf(i), yOf(signal[i])),
    ];

    // Línea que une un pico R con el siguiente = intervalo R-R.
    if (points.length > 1) {
      final rrLine = Paint()
        ..color = _rrLine
        ..strokeWidth = 1.4
        ..style = PaintingStyle.stroke;
      final path = Path()..moveTo(points.first.dx, points.first.dy);
      for (var k = 1; k < points.length; k++) {
        path.lineTo(points[k].dx, points[k].dy);
      }
      canvas.drawPath(path, rrLine);
    }

    // Círculo sobre cada pico R (punto claro/oscuro según tema + anillo rojo).
    final dot = Paint()..color = _rrDot;
    final ring = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = const Color(0xFFEF4444);
    for (final o in points) {
      canvas.drawCircle(o, 3.2, dot);
      canvas.drawCircle(o, 5.0, ring);
    }
  }

  /// Medical-standard ECG grid: 1 mm minor squares with darker major lines
  /// every 5 mm horizontally and every 10 mm vertically.
  void _drawEcgGrid(Canvas canvas, Size size) {
    final mm = size.height / 40.0; // px per millimetre (≈40 mm tall)
    if (mm <= 0) return;

    // Sobre blanco la rejilla se sube de opacidad para lograr el rosa del papel
    // de ECG; sobre el fondo oscuro se mantiene tenue para no tapar el trazado.
    final minor = Paint()
      ..color = isDark ? const Color(0x12EF4444) : const Color(0x26EF4444)
      ..strokeWidth = 1;
    final major = Paint()
      ..color = isDark ? const Color(0x40EF4444) : const Color(0x66EF4444)
      ..strokeWidth = 1.2;

    // Vertical lines (columns): minor each 1 mm, major each 5 mm.
    final cols = (size.width / mm).ceil();
    for (var c = 0; c <= cols; c++) {
      final x = c * mm;
      canvas.drawLine(
          Offset(x, 0), Offset(x, size.height), c % 5 == 0 ? major : minor);
    }
    // Horizontal lines (rows): minor each 1 mm, major each 10 mm.
    final rows = (size.height / mm).ceil();
    for (var r = 0; r <= rows; r++) {
      final y = r * mm;
      canvas.drawLine(
          Offset(0, y), Offset(size.width, y), r % 10 == 0 ? major : minor);
    }
  }

  @override
  bool shouldRepaint(covariant _GradCamPainter old) =>
      old.signal != signal ||
      old.importance != importance ||
      // Sin esto el lienzo no se redibuja al alternar claro/oscuro, porque la
      // señal es el mismo objeto y el painter se daría por no cambiado.
      old.isDark != isDark;
}
