import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'config.dart';
import 'theme.dart';

/// A file the user picked to analyse: its bytes and original name.
class PickedEcg {
  final Uint8List bytes;
  final String name;
  PickedEcg(this.bytes, this.name);
}

/// Shows a bottom sheet offering the ways to bring in an ECG (import a data file,
/// take a photo, or pick from the gallery) and returns the chosen file, or null
/// if the user cancelled.
Future<PickedEcg?> pickEcgSource(BuildContext context) async {
  final showCamera = !kIsWeb &&
      (defaultTargetPlatform == TargetPlatform.android ||
          defaultTargetPlatform == TargetPlatform.iOS);

  return showModalBottomSheet<PickedEcg>(
    context: context,
    backgroundColor: AppColors.surface,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
    ),
    builder: (sheetCtx) {
      Future<void> done(Future<PickedEcg?> Function() run) async {
        try {
          final picked = await run();
          if (sheetCtx.mounted) Navigator.pop(sheetCtx, picked);
        } catch (e) {
          if (sheetCtx.mounted) Navigator.pop(sheetCtx);
          if (context.mounted) {
            ScaffoldMessenger.of(context)
                .showSnackBar(SnackBar(content: Text('$e')));
          }
        }
      }

      return SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 12),
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: AppColors.border,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 8),
            const Padding(
              padding: EdgeInsets.fromLTRB(20, 10, 20, 4),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text('Agregar ECG',
                    style: TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 16,
                        color: AppColors.text)),
              ),
            ),
            _SourceTile(
              icon: Icons.upload_file_rounded,
              title: 'Importar archivo',
              subtitle: '.pt · .mat · .dat · imagen · PDF',
              onTap: () => done(_pickFile),
            ),
            if (showCamera)
              _SourceTile(
                icon: Icons.photo_camera_rounded,
                title: 'Tomar una foto',
                subtitle: 'Capturar un trazado o monitor con la cámara',
                onTap: () => done(() => _pickImage(ImageSource.camera)),
              ),
            _SourceTile(
              icon: Icons.image_outlined,
              title: 'Elegir una imagen',
              subtitle: 'Seleccionar una foto de ECG de la galería',
              onTap: () => done(() => _pickImage(ImageSource.gallery)),
            ),
            const SizedBox(height: 12),
          ],
        ),
      );
    },
  );
}

Future<PickedEcg?> _pickFile() async {
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

Future<PickedEcg?> _pickImage(ImageSource source) async {
  final picker = ImagePicker();
  final XFile? shot = await picker.pickImage(source: source, imageQuality: 95);
  if (shot == null) return null;
  final bytes = await shot.readAsBytes();
  return PickedEcg(bytes, shot.name);
}

class _SourceTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _SourceTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Container(
        padding: const EdgeInsets.all(9),
        decoration: BoxDecoration(
          color: const Color(0x2638BDF8),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(icon, color: AppColors.accent),
      ),
      title: Text(title,
          style: const TextStyle(
              color: AppColors.text, fontWeight: FontWeight.w600, fontSize: 14.5)),
      subtitle: Text(subtitle,
          style: const TextStyle(color: AppColors.muted, fontSize: 12)),
      onTap: onTap,
    );
  }
}
