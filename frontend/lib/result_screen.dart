import 'package:flutter/material.dart';

import 'models.dart';
import 'theme.dart';
import 'widgets/gradcam_ecg.dart';
import 'widgets/probability_bars.dart';

/// Full breakdown of a single prediction.
class ResultScreen extends StatelessWidget {
  final PredictionResult result;
  const ResultScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final leadSignal =
        result.allLeads[result.gradcamLead] ?? result.allLeads.values.firstOrNull ?? const [];

    return Scaffold(
      appBar: AppBar(title: const Text('Analysis result')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 40),
        children: [
          _VerdictCard(result: result),
          if (result.warning != null) ...[
            const SizedBox(height: 12),
            _WarningBanner(text: result.warning!),
          ],
          const SizedBox(height: 16),
          _SectionCard(
            title: 'Where the model looked (Grad-CAM)',
            subtitle:
                'The trace is your ECG. Hotter colours mark the moments that '
                'most influenced the verdict.',
            child: GradCamEcg(
              signal: leadSignal,
              importance: result.gradcam,
              leadLabel: result.gradcamLead,
            ),
          ),
          const SizedBox(height: 16),
          _SectionCard(
            title: 'Probabilities across all classes',
            subtitle: 'What else the model considered, and how strongly.',
            child: ProbabilityBars(result: result),
          ),
          const SizedBox(height: 16),
          _SectionCard(
            title: 'Why this result',
            subtitle:
                'Features characteristic of ${result.fullName} — what a reading '
                'of this rhythm is based on.',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(result.description,
                    style: const TextStyle(
                        color: AppColors.text, height: 1.5, fontSize: 14)),
                if (result.keyFeatures.isNotEmpty) ...[
                  const SizedBox(height: 14),
                  for (final f in result.keyFeatures) _FeatureRow(text: f),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _VerdictCard extends StatelessWidget {
  final PredictionResult result;
  const _VerdictCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 14,
                  height: 14,
                  decoration: BoxDecoration(
                      color: result.color, shape: BoxShape.circle),
                ),
                const SizedBox(width: 10),
                Text(
                  result.prediction,
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.w800,
                    color: result.color,
                    letterSpacing: -0.5,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(result.fullName,
                style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: AppColors.text)),
            const SizedBox(height: 16),
            Row(
              children: [
                const Text('Confidence',
                    style: TextStyle(color: AppColors.muted)),
                const Spacer(),
                Text('${(result.confidence * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                        color: result.color, fontWeight: FontWeight.w700)),
              ],
            ),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                value: result.confidence.clamp(0.0, 1.0),
                minHeight: 10,
                backgroundColor: AppColors.surface2,
                valueColor: AlwaysStoppedAnimation(result.color),
              ),
            ),
            if (result.filename.isNotEmpty) ...[
              const SizedBox(height: 14),
              Text(result.filename,
                  style: const TextStyle(
                      color: AppColors.muted,
                      fontSize: 12,
                      fontFamily: 'monospace')),
            ],
          ],
        ),
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final Widget child;
  const _SectionCard(
      {required this.title, required this.subtitle, required this.child});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.text)),
            const SizedBox(height: 4),
            Text(subtitle,
                style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
            const SizedBox(height: 16),
            child,
          ],
        ),
      ),
    );
  }
}

class _FeatureRow extends StatelessWidget {
  final String text;
  const _FeatureRow({required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 3, right: 10),
            child: Icon(Icons.monitor_heart_outlined,
                size: 16, color: AppColors.accent),
          ),
          Expanded(
            child: Text(text,
                style: const TextStyle(
                    color: AppColors.text, height: 1.4, fontSize: 13.5)),
          ),
        ],
      ),
    );
  }
}

class _WarningBanner extends StatelessWidget {
  final String text;
  const _WarningBanner({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0x22F59E0B),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0x55F59E0B)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.warning_amber_rounded,
              color: Color(0xFFF59E0B), size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text,
                style: const TextStyle(
                    color: Color(0xFFFBBF24), fontSize: 12.5, height: 1.4)),
          ),
        ],
      ),
    );
  }
}

extension _FirstOrNull<E> on Iterable<E> {
  E? get firstOrNull => isEmpty ? null : first;
}
