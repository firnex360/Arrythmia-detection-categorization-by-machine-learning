import 'package:flutter/material.dart';

import 'package:frontend/models/models.dart';
import 'package:frontend/core/theme.dart';

/// Horizontal bars showing the probability the model assigned to every
/// arrhythmia class — so the user sees not just the verdict but also which
/// other rhythms were "in the running".
class ProbabilityBars extends StatelessWidget {
  final PredictionResult result;
  const ProbabilityBars({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final entries = result.sortedProbs;
    return Column(
      children: [
        for (final e in entries) _bar(e.key, e.value),
      ],
    );
  }

  Widget _bar(String code, double p) {
    final isTop = code == result.prediction;
    final color = result.classColors[code] ?? AppColors.accent;
    final name = result.classNames[code] ?? code;

    return Padding(
      padding: EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 10,
                height: 10,
                decoration:
                    BoxDecoration(color: color, shape: BoxShape.circle),
              ),
              SizedBox(width: 8),
              Text(
                code,
                style: TextStyle(
                  fontWeight: isTop ? FontWeight.w700 : FontWeight.w500,
                  color: AppColors.text,
                ),
              ),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  name,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(color: AppColors.muted, fontSize: 12),
                ),
              ),
              Text(
                '${(p * 100).toStringAsFixed(1)}%',
                style: TextStyle(
                  fontFeatures: const [FontFeature.tabularFigures()],
                  fontWeight: isTop ? FontWeight.w700 : FontWeight.w500,
                  color: isTop ? color : AppColors.text,
                ),
              ),
            ],
          ),
          SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: p.clamp(0.0, 1.0),
              minHeight: 8,
              backgroundColor: AppColors.surface2,
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
        ],
      ),
    );
  }
}
