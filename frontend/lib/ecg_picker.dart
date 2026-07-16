import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';

import 'config.dart';

/// A file the user picked to analyse: its bytes and original name.
class PickedEcg {
  final Uint8List bytes;
  final String name;
  PickedEcg(this.bytes, this.name);
}

/// Opens the system file picker for an ECG data file (.pt / .mat / .dat).
/// Returns the chosen file, or null if cancelled. Images are no longer supported.
Future<PickedEcg?> pickEcgFile() async {
  final res = await FilePicker.platform.pickFiles(
    type: FileType.custom,
    allowedExtensions: AppConfig.allExtensions,
    withData: true,
  );
  if (res == null || res.files.isEmpty) return null;
  final f = res.files.first;
  final bytes = f.bytes;
  if (bytes == null) throw 'No se pudo leer el archivo seleccionado.';
  return PickedEcg(bytes, f.name);
}
