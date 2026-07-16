import 'package:flutter/material.dart';

import '../theme.dart';

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
              color: const Color(0xFF0A0E17),
              child: CustomPaint(
                painter: _GradCamPainter(signal: signal, importance: importance),
                size: Size.infinite,
              ),
            ),
          ),
        ),
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

  _GradCamPainter({required this.signal, required this.importance});

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
    _drawGrid(canvas, size);
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
  }

  void _drawGrid(Canvas canvas, Size size) {
    final grid = Paint()
      ..color = const Color(0x11EF4444)
      ..strokeWidth = 1;
    const cells = 20;
    for (var i = 0; i <= cells; i++) {
      final x = size.width * i / cells;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), grid);
    }
    for (var i = 0; i <= cells ~/ 2; i++) {
      final y = size.height * i / (cells ~/ 2);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), grid);
    }
  }

  @override
  bool shouldRepaint(covariant _GradCamPainter old) =>
      old.signal != signal || old.importance != importance;
}
