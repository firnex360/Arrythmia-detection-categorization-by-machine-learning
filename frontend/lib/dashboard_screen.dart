import 'package:flutter/material.dart';

import 'api_service.dart';
import 'models.dart';
import 'theme.dart';
import 'widgets/charts.dart';

/// Global analytics across every ECG in the system: distribution of arrhythmias,
/// how they break down by age and gender, activity over time, and how accurate
/// the model has been according to the doctors' verdicts.
class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  DashboardData? _data;
  String? _error;
  String _granularity = 'day'; // 'day' | 'hour' for the timeline chart

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _error = null);
    try {
      final d = await ApiService.dashboard();
      if (!mounted) return;
      setState(() => _data = d);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Color _colorFor(String code) =>
      _data?.classColors[code] ?? arrhythmiaColor(code);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Dashboard Poblacional')),
      body: RefreshIndicator(onRefresh: _load, child: _body()),
    );
  }

  Widget _body() {
    if (_error != null) {
      return ListView(children: [
        const SizedBox(height: 80),
        const Icon(Icons.cloud_off_rounded, size: 48, color: AppColors.muted),
        const SizedBox(height: 12),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Text(_error!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: AppColors.muted)),
        ),
        const SizedBox(height: 16),
        Center(
          child: OutlinedButton.icon(
              onPressed: _load,
              icon: const Icon(Icons.refresh),
              label: const Text('Reintentar')),
        ),
      ]);
    }
    final d = _data;
    if (d == null) return const Center(child: CircularProgressIndicator());

    final present = <String>[
      ...d.classOrder.where(d.byClass.containsKey),
      ...d.byClass.keys.where((c) => !d.classOrder.contains(c)),
    ];

    if (d.totalRecords == 0) {
      return ListView(children: const [
        SizedBox(height: 90),
        Icon(Icons.insights_outlined, size: 54, color: AppColors.muted),
        SizedBox(height: 14),
        Padding(
          padding: EdgeInsets.symmetric(horizontal: 32),
          child: Text(
            'Aún no hay ECG analizados en el sistema.\n'
            'Cuando agregues ECG a tus pacientes, aquí verás las tendencias por '
            'edad, género, tipo de arritmia y la precisión del modelo.',
            textAlign: TextAlign.center,
            style: TextStyle(color: AppColors.muted, height: 1.5),
          ),
        ),
      ]);
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final w = constraints.maxWidth;
        final cols = w >= 1150 ? 3 : (w >= 760 ? 2 : 1);
        final panels = _panels(d, present);

        return ListView(
          padding: EdgeInsets.symmetric(
              horizontal: w >= 900 ? 32 : 16, vertical: 18),
          children: [
            Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1240),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Resumen del programa',
                        style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                            color: AppColors.text)),
                    const SizedBox(height: 2),
                    const Text(
                      'Datos agregados de todos los ECG registrados en el sistema.',
                      style: TextStyle(color: AppColors.muted, fontSize: 13),
                    ),
                    const SizedBox(height: 16),
                    _statRow(d),
                    const SizedBox(height: 18),
                    _masonry(panels, cols),
                    const SizedBox(height: 18),
                    _Panel(
                      title: 'Leyenda de arritmias',
                      subtitle:
                          'Códigos y colores usados en todos los gráficos.',
                      child: _Legend(
                          classes: present,
                          names: d.classNames,
                          colorFor: _colorFor),
                    ),
                  ],
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _statRow(DashboardData d) {
    final acc = d.accuracy.accuracy;
    return Wrap(
      spacing: 12,
      runSpacing: 12,
      children: [
        _StatTile(
            label: 'Pacientes',
            value: '${d.totalPatients}',
            icon: Icons.people_alt_outlined),
        _StatTile(
            label: 'ECG analizados',
            value: '${d.totalRecords}',
            icon: Icons.monitor_heart_outlined),
        _StatTile(
            label: 'Doctores',
            value: '${d.totalDoctors}',
            icon: Icons.medical_services_outlined),
        _StatTile(
          label: 'Precisión del modelo',
          value: acc == null ? '—' : '${(acc * 100).toStringAsFixed(0)}%',
          icon: Icons.verified_outlined,
          highlight: true,
        ),
      ],
    );
  }

  List<Widget> _panels(DashboardData d, List<String> present) {
    final maxClass = d.byClass.values.fold<int>(0, (m, v) => v > m ? v : m);

    return [
      _Panel(
        title: 'Distribución de arritmias',
        subtitle: 'Proporción de cada tipo sobre el total de ECG.',
        child: DonutChart(
          centerLabel: 'ECG',
          slices: [
            for (final code in present)
              PieSlice(code, (d.byClass[code] ?? 0).toDouble(),
                  _colorFor(code)),
          ],
        ),
      ),
      _Panel(
        title: 'ECG a lo largo del tiempo',
        subtitle: 'Volumen de ECG por ${_granularity == 'hour' ? 'hora' : 'día'}, '
            'segmentado por tipo de arritmia.',
        child: _TimelineSection(
          day: d.timelineDay,
          hour: d.timelineHour,
          granularity: _granularity,
          onGranularity: (g) => setState(() => _granularity = g),
          classes: present,
          colorFor: _colorFor,
        ),
      ),
      _AccuracyPanel(
        accuracy: d.accuracy,
        classOrder: present,
        classNames: d.classNames,
        colorFor: _colorFor,
      ),
      _Panel(
        title: 'Conteo por arritmia',
        subtitle: 'Número total de ECG de cada tipo.',
        child: Column(
          children: [
            for (final code in present)
              _CountBar(
                label: '$code · ${d.classNames[code] ?? code}',
                value: d.byClass[code] ?? 0,
                max: maxClass,
                color: _colorFor(code),
              ),
          ],
        ),
      ),
      _Panel(
        title: 'Arritmias por rango de edad',
        subtitle:
            '¿Qué grupos de edad son más propensos a cada arritmia? Cada barra '
            'está segmentada por tipo.',
        child: _StackedGroups(
          groups: d.byAge,
          classes: present,
          colorFor: _colorFor,
          emptyText: 'Sin datos de edad (faltan fechas de nacimiento).',
        ),
      ),
      _Panel(
        title: 'Arritmias por género',
        subtitle: '¿Predomina alguna arritmia en un género frente al otro?',
        child: _StackedGroups(
          groups: d.byGender,
          classes: present,
          colorFor: _colorFor,
          labelMap: const {
            'F': 'Femenino',
            'M': 'Masculino',
            'Other': 'Otro',
            'Unknown': 'Sin dato',
          },
          emptyText: 'Sin datos de género.',
        ),
      ),
    ];
  }

  /// Distribute [panels] into [cols] balanced columns (masonry).
  Widget _masonry(List<Widget> panels, int cols) {
    if (cols <= 1) {
      return Column(
        children: [
          for (final p in panels)
            Padding(padding: const EdgeInsets.only(bottom: 16), child: p),
        ],
      );
    }
    final columns = List.generate(cols, (_) => <Widget>[]);
    for (var i = 0; i < panels.length; i++) {
      columns[i % cols].add(panels[i]);
    }
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (var i = 0; i < cols; i++) ...[
          Expanded(
            child: Column(
              children: [
                for (final p in columns[i])
                  Padding(
                      padding: const EdgeInsets.only(bottom: 16), child: p),
              ],
            ),
          ),
          if (i < cols - 1) const SizedBox(width: 16),
        ],
      ],
    );
  }
}

/// Timeline chart with a Horas/Días granularity switch.
class _TimelineSection extends StatelessWidget {
  final List<TimelinePoint> day;
  final List<TimelinePoint> hour;
  final String granularity;
  final ValueChanged<String> onGranularity;
  final List<String> classes;
  final Color Function(String) colorFor;

  const _TimelineSection({
    required this.day,
    required this.hour,
    required this.granularity,
    required this.onGranularity,
    required this.classes,
    required this.colorFor,
  });

  String _fmt(String date) {
    // day: 'YYYY-MM-DD' -> 'MM-DD' ; hour: 'YYYY-MM-DDTHH' -> 'DD HH:00'
    if (date.contains('T')) {
      final parts = date.split('T');
      final d = parts[0].length >= 10 ? parts[0].substring(8) : parts[0];
      return '$d ${parts.length > 1 ? parts[1] : ''}:00';
    }
    return date.length >= 10 ? date.substring(5) : date;
  }

  @override
  Widget build(BuildContext context) {
    final series = granularity == 'hour' ? hour : day;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Align(
          alignment: Alignment.centerRight,
          child: SegmentedButton<String>(
            segments: const [
              ButtonSegment(
                  value: 'hour',
                  label: Text('Horas'),
                  icon: Icon(Icons.schedule, size: 16)),
              ButtonSegment(
                  value: 'day',
                  label: Text('Días'),
                  icon: Icon(Icons.calendar_today, size: 15)),
            ],
            selected: {granularity},
            onSelectionChanged: (s) => onGranularity(s.first),
            style: ButtonStyle(
              visualDensity: VisualDensity.compact,
              textStyle: WidgetStateProperty.all(const TextStyle(fontSize: 12)),
            ),
          ),
        ),
        const SizedBox(height: 12),
        if (series.isEmpty)
          const _Empty('Sin actividad en esta granularidad.')
        else
          SimpleLineChart(
            xLabels: [for (final p in series) _fmt(p.date)],
            series: [
              for (final code in classes)
                LineSeries(
                  code,
                  colorFor(code),
                  [for (final p in series) (p.counts[code] ?? 0).toDouble()],
                ),
            ],
          ),
      ],
    );
  }
}

class _StatTile extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final bool highlight;
  const _StatTile({
    required this.label,
    required this.value,
    required this.icon,
    this.highlight = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 190,
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
      decoration: BoxDecoration(
        color: highlight ? const Color(0x1A38BDF8) : AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
            color: highlight ? const Color(0x5538BDF8) : AppColors.border),
      ),
      child: Row(
        children: [
          Icon(icon, color: AppColors.accent, size: 26),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(value,
                  style: const TextStyle(
                      color: AppColors.text,
                      fontSize: 22,
                      fontWeight: FontWeight.w800)),
              Text(label,
                  style: const TextStyle(color: AppColors.muted, fontSize: 11)),
            ],
          ),
        ],
      ),
    );
  }
}

class _Panel extends StatelessWidget {
  final String title;
  final String subtitle;
  final Widget child;
  const _Panel(
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

/// Model accuracy from the doctors' verdicts: overall + per-class + confusions.
class _AccuracyPanel extends StatelessWidget {
  final AccuracyStats accuracy;
  final List<String> classOrder;
  final Map<String, String> classNames;
  final Color Function(String) colorFor;

  const _AccuracyPanel({
    required this.accuracy,
    required this.classOrder,
    required this.classNames,
    required this.colorFor,
  });

  @override
  Widget build(BuildContext context) {
    if (accuracy.reviewed == 0) {
      return _Panel(
        title: 'Precisión del modelo',
        subtitle: 'Según lo que confirmen los doctores en cada ECG.',
        child: Row(
          children: const [
            Icon(Icons.rule_rounded, color: AppColors.muted, size: 22),
            SizedBox(width: 10),
            Expanded(
              child: Text(
                'Aún no hay ECG revisados. En cada resultado, marca si el modelo '
                'acertó (Correcto / Incorrecto) y aquí verás qué arritmias predice mejor.',
                style: TextStyle(color: AppColors.muted, height: 1.4, fontSize: 12.5),
              ),
            ),
          ],
        ),
      );
    }

    final pct = ((accuracy.accuracy ?? 0) * 100).toStringAsFixed(0);
    final classes = classOrder.where(accuracy.byClass.containsKey).toList();

    return _Panel(
      title: 'Precisión del modelo',
      subtitle:
          '${accuracy.correct} de ${accuracy.reviewed} revisados correctos'
          '${accuracy.unreviewed > 0 ? ' · ${accuracy.unreviewed} sin revisar' : ''}.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('$pct%',
                  style: const TextStyle(
                      color: AppColors.accent,
                      fontSize: 30,
                      fontWeight: FontWeight.w800)),
              const SizedBox(width: 8),
              const Padding(
                padding: EdgeInsets.only(top: 10),
                child: Text('acierto global',
                    style: TextStyle(color: AppColors.muted, fontSize: 12)),
              ),
            ],
          ),
          const SizedBox(height: 10),
          const Text('Acierto por tipo de arritmia',
              style: TextStyle(
                  color: AppColors.text,
                  fontSize: 13,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          for (final code in classes)
            _AccuracyBar(
              code: code,
              stat: accuracy.byClass[code]!,
              color: colorFor(code),
            ),
          if (accuracy.confusion.isNotEmpty) ...[
            const SizedBox(height: 12),
            const Text('Confusiones más comunes',
                style: TextStyle(
                    color: AppColors.text,
                    fontSize: 13,
                    fontWeight: FontWeight.w600)),
            const SizedBox(height: 6),
            for (final e in (accuracy.confusion.entries.toList()
                  ..sort((a, b) => b.value.compareTo(a.value))))
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 2),
                child: Row(
                  children: [
                    const Icon(Icons.swap_horiz_rounded,
                        size: 15, color: Color(0xFFEF4444)),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        _confusionLabel(e.key),
                        style: const TextStyle(
                            color: AppColors.text, fontSize: 12.5),
                      ),
                    ),
                    Text('${e.value}',
                        style: const TextStyle(
                            color: AppColors.muted, fontSize: 12)),
                  ],
                ),
              ),
          ],
        ],
      ),
    );
  }

  String _confusionLabel(String key) {
    final parts = key.split('→');
    if (parts.length != 2) return key;
    return 'Predijo ${parts[0]}, era ${parts[1]}';
  }
}

class _AccuracyBar extends StatelessWidget {
  final String code;
  final ClassAccuracy stat;
  final Color color;
  const _AccuracyBar(
      {required this.code, required this.stat, required this.color});

  @override
  Widget build(BuildContext context) {
    final frac = stat.accuracy ?? 0;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(code,
                    style:
                        const TextStyle(color: AppColors.text, fontSize: 13)),
              ),
              Text('${stat.correct}/${stat.reviewed}  ·  '
                  '${(frac * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(
                      color: AppColors.muted, fontSize: 12)),
            ],
          ),
          const SizedBox(height: 5),
          ClipRRect(
            borderRadius: BorderRadius.circular(5),
            child: LinearProgressIndicator(
              value: frac.clamp(0.0, 1.0),
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

/// A single labelled horizontal bar scaled to [max].
class _CountBar extends StatelessWidget {
  final String label;
  final int value;
  final int max;
  final Color color;
  const _CountBar(
      {required this.label,
      required this.value,
      required this.max,
      required this.color});

  @override
  Widget build(BuildContext context) {
    final frac = max <= 0 ? 0.0 : value / max;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(label,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style:
                        const TextStyle(color: AppColors.text, fontSize: 13)),
              ),
              Text('$value',
                  style: TextStyle(
                      color: color, fontWeight: FontWeight.w700, fontSize: 13)),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(5),
            child: LinearProgressIndicator(
              value: frac.clamp(0.0, 1.0),
              minHeight: 9,
              backgroundColor: AppColors.surface2,
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
        ],
      ),
    );
  }
}

/// One stacked horizontal bar per group (age band or gender).
class _StackedGroups extends StatelessWidget {
  final List<GroupCounts> groups;
  final List<String> classes;
  final Color Function(String) colorFor;
  final Map<String, String>? labelMap;
  final String emptyText;

  const _StackedGroups({
    required this.groups,
    required this.classes,
    required this.colorFor,
    required this.emptyText,
    this.labelMap,
  });

  @override
  Widget build(BuildContext context) {
    if (groups.isEmpty) return _Empty(emptyText);

    return Column(
      children: [
        for (final g in groups)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 7),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(labelMap?[g.label] ?? g.label,
                          style: const TextStyle(
                              color: AppColors.text,
                              fontSize: 13,
                              fontWeight: FontWeight.w600)),
                    ),
                    Text('${g.total}',
                        style: const TextStyle(
                            color: AppColors.muted, fontSize: 12)),
                  ],
                ),
                const SizedBox(height: 6),
                ClipRRect(
                  borderRadius: BorderRadius.circular(5),
                  child: SizedBox(
                    height: 14,
                    child: Row(
                      children: [
                        for (final code in classes)
                          if ((g.counts[code] ?? 0) > 0)
                            Expanded(
                              flex: g.counts[code]!,
                              child: Container(color: colorFor(code)),
                            ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}

class _Legend extends StatelessWidget {
  final List<String> classes;
  final Map<String, String> names;
  final Color Function(String) colorFor;
  const _Legend(
      {required this.classes, required this.names, required this.colorFor});

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 16,
      runSpacing: 8,
      children: [
        for (final code in classes)
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                    color: colorFor(code),
                    borderRadius: BorderRadius.circular(3)),
              ),
              const SizedBox(width: 6),
              Text('$code · ${names[code] ?? code}',
                  style: const TextStyle(color: AppColors.muted, fontSize: 12)),
            ],
          ),
      ],
    );
  }
}

class _Empty extends StatelessWidget {
  final String text;
  const _Empty(this.text);
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Text(text,
          style: const TextStyle(color: AppColors.muted, fontSize: 12.5)),
    );
  }
}
