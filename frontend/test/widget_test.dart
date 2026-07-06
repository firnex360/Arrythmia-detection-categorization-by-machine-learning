// Basic smoke test: the app boots and shows the home screen.

import 'package:flutter_test/flutter_test.dart';

import 'package:frontend/main.dart';

void main() {
  testWidgets('App shows the home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const EcgApp());

    expect(find.text('Arrhythmia analysis'), findsOneWidget);
    expect(find.text('Import ECG file'), findsOneWidget);
  });
}
