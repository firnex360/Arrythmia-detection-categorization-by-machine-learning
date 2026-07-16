import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme.dart';

/// A slice for the donut chart.
class PieSlice {
  final String label;
  final double value;
  final Color color;
  const PieSlice(this.label, this.value, this.color);
}

/// Donut/pie chart with a legend showing each slice's share.
class DonutChart extends StatelessWidget {
  final List<PieSlice> slices;
  final String centerLabel;
  final double size;

  const DonutChart({
    super.key,
    required this.slices,
    this.centerLabel = '',
    this.size = 160,
  });

  @override
  Widget build(BuildContext context) {
    final total = slices.fold<double>(0, (s, e) => s + e.value);
    final donut = SizedBox(
      width: size,
      height: size,
      child: CustomPaint(
        painter: _DonutPainter(slices, total),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(total.toStringAsFixed(0),
                  style: TextStyle(
                      color: AppColors.text,
                      fontSize: 26,
                      fontWeight: FontWeight.w800)),
              if (centerLabel.isNotEmpty)
                Text(centerLabel,
                    style: TextStyle(color: AppColors.muted, fontSize: 11)),
            ],
          ),
        ),
      ),
    );

    final legend = Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (final s in slices)
          Padding(
            padding: EdgeInsets.symmetric(vertical: 4),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                      color: s.color, borderRadius: BorderRadius.circular(3)),
                ),
                SizedBox(width: 8),
                Text(s.label,
                    style: TextStyle(color: AppColors.text, fontSize: 13)),
                SizedBox(width: 6),
                Text(
                  total <= 0
                      ? '0%'
                      : '${(s.value / total * 100).toStringAsFixed(0)}%',
                  style: TextStyle(
                      color: AppColors.muted,
                      fontSize: 12,
                      fontWeight: FontWeight.w600),
                ),
              ],
            ),
          ),
      ],
    );

    return LayoutBuilder(
      builder: (context, c) {
        if (c.maxWidth < 260) {
          return Column(
            children: [donut, SizedBox(height: 12), legend],
          );
        }
        return Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            donut,
            SizedBox(width: 24),
            Flexible(child: legend),
          ],
        );
      },
    );
  }
}

class _DonutPainter extends CustomPainter {
  final List<PieSlice> slices;
  final double total;
  _DonutPainter(this.slices, this.total);

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final stroke = size.shortestSide * 0.20;
    final radius = (size.shortestSide - stroke) / 2;
    final center = rect.center;

    final bg = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = stroke
      ..color = AppColors.surface2;
    canvas.drawCircle(center, radius, bg);

    if (total <= 0) return;

    var start = -math.pi / 2;
    for (final s in slices) {
      if (s.value <= 0) continue;
      final sweep = s.value / total * 2 * math.pi;
      final p = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = stroke
        ..strokeCap = StrokeCap.butt
        ..color = s.color;
      canvas.drawArc(
        Rect.fromCircle(center: center, radius: radius),
        start,
        sweep - 0.03, // tiny gap between slices
        false,
        p,
      );
      start += sweep;
    }
  }

  @override
  bool shouldRepaint(covariant _DonutPainter old) =>
      old.slices != slices || old.total != total;
}

/// A named line series for the line chart.
class LineSeries {
  final String label;
  final Color color;
  final List<double> values; // one value per x position
  const LineSeries(this.label, this.color, this.values);
}

/// Multi-series line chart with light grid, used for the ECG-over-time timeline.
class SimpleLineChart extends StatelessWidget {
  final List<String> xLabels;
  final List<LineSeries> series;
  final double height;

  const SimpleLineChart({
    super.key,
    required this.xLabels,
    required this.series,
    this.height = 200,
  });

  @override
  Widget build(BuildContext context) {
    double maxY = 0;
    for (final s in series) {
      for (final v in s.values) {
        if (v > maxY) maxY = v;
      }
    }
    if (maxY <= 0) maxY = 1;

    return SizedBox(
      height: height,
      child: CustomPaint(
        painter: _LinePainter(xLabels: xLabels, series: series, maxY: maxY),
        child: SizedBox.expand(),
      ),
    );
  }
}

class _LinePainter extends CustomPainter {
  final List<String> xLabels;
  final List<LineSeries> series;
  final double maxY;
  _LinePainter({required this.xLabels, required this.series, required this.maxY});

  @override
  void paint(Canvas canvas, Size size) {
    const leftPad = 28.0;
    const bottomPad = 22.0;
    const topPad = 8.0;
    final plot = Rect.fromLTRB(leftPad, topPad, size.width, size.height - bottomPad);

    final gridPaint = Paint()
      ..color = AppColors.surface2
      ..strokeWidth = 1;
    final textStyle = TextStyle(color: AppColors.muted, fontSize: 10);

    // Horizontal grid + y labels (0 .. maxY in a few steps).
    final steps = maxY <= 4 ? maxY.toInt() : 4;
    final safeSteps = steps <= 0 ? 1 : steps;
    for (var i = 0; i <= safeSteps; i++) {
      final t = i / safeSteps;
      final y = plot.bottom - t * plot.height;
      canvas.drawLine(Offset(plot.left, y), Offset(plot.right, y), gridPaint);
      final val = (maxY * t).round();
      _text(canvas, '$val', Offset(0, y - 6), textStyle);
    }

    final n = xLabels.length;
    double xAt(int i) =>
        n <= 1 ? plot.center.dx : plot.left + (i / (n - 1)) * plot.width;
    double yAt(double v) => plot.bottom - (v / maxY) * plot.height;

    for (final s in series) {
      final linePaint = Paint()
        ..color = s.color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5
        ..strokeJoin = StrokeJoin.round
        ..strokeCap = StrokeCap.round;
      final dotPaint = Paint()..color = s.color;

      final path = Path();
      for (var i = 0; i < s.values.length; i++) {
        final o = Offset(xAt(i), yAt(s.values[i]));
        if (i == 0) {
          path.moveTo(o.dx, o.dy);
        } else {
          path.lineTo(o.dx, o.dy);
        }
        if (n <= 8) canvas.drawCircle(o, 3, dotPaint);
      }
      canvas.drawPath(path, linePaint);
    }

    // X labels: first and last (avoids clutter).
    if (n > 0) {
      _text(canvas, xLabels.first, Offset(plot.left, size.height - 14), textStyle);
      if (n > 1) {
        final tp = _painter(xLabels.last, textStyle);
        _text(canvas, xLabels.last,
            Offset(plot.right - tp.width, size.height - 14), textStyle);
      }
    }
  }

  TextPainter _painter(String s, TextStyle style) {
    final tp = TextPainter(
        text: TextSpan(text: s, style: style),
        textDirection: TextDirection.ltr)
      ..layout();
    return tp;
  }

  void _text(Canvas canvas, String s, Offset at, TextStyle style) {
    _painter(s, style).paint(canvas, at);
  }

  @override
  bool shouldRepaint(covariant _LinePainter old) =>
      old.series != series || old.maxY != maxY || old.xLabels != xLabels;
}
