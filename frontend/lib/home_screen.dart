import 'package:desktop_drop/desktop_drop.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'api_service.dart';
import 'config.dart';
import 'result_screen.dart';
import 'theme.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _busy = false;
  String _status = '';

  Future<void> _run(Future<_PickedFile?> Function() pick) async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _status = 'Reading file…';
    });
    try {
      final file = await pick();
      if (file == null) {
        setState(() {
          _busy = false;
          _status = '';
        });
        return;
      }
      setState(() => _status = 'Running the model on ${file.name}…');
      final result =
          await ApiService.predict(bytes: file.bytes, filename: file.name);
      if (!mounted) return;
      setState(() {
        _busy = false;
        _status = '';
      });
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _status = '';
      });
      _showError('$e');
    }
  }

  /// Import a data file (.pt / .mat / .dat) or an image from storage.
  Future<_PickedFile?> _pickFile() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: AppConfig.allExtensions,
      withData: true, // ensures bytes are available on every platform
    );
    if (res == null || res.files.isEmpty) return null;
    final f = res.files.first;
    final bytes = f.bytes;
    if (bytes == null) throw 'Could not read the selected file.';
    return _PickedFile(bytes, f.name);
  }

  /// Capture an ECG photo with the camera, or pick one from the gallery.
  Future<_PickedFile?> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final XFile? shot = await picker.pickImage(source: source, imageQuality: 95);
    if (shot == null) return null;
    final bytes = await shot.readAsBytes();
    return _PickedFile(bytes, shot.name);
  }

  Future<void> _handleDrop(DropDoneDetails details) async {
    if (_busy) return;
    if (details.files.isEmpty) return;

    final xfile = details.files.first;
    
    setState(() {
      _busy = true;
      _status = 'Reading dropped file…';
    });

    try {
      final bytes = await xfile.readAsBytes();
      final picked = _PickedFile(bytes, xfile.name);
      
      setState(() => _status = 'Running the model on ${picked.name}…');
      final result = await ApiService.predict(bytes: picked.bytes, filename: picked.name);
      
      if (!mounted) return;
      setState(() {
        _busy = false;
        _status = '';
      });
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _status = '';
      });
      _showError('$e');
    }
  }

  void _showError(String message) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: const Text('Something went wrong'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Future<void> _editServerUrl() async {
    final controller = TextEditingController(text: AppConfig.baseUrl);
    final url = await showDialog<String>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: const Text('Backend server URL'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Where the Python/Flask backend (app.py) is running. On a real '
              'phone use your computer\'s LAN IP, e.g. http://192.168.1.20:5000',
              style: TextStyle(color: AppColors.muted, fontSize: 12.5),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'http://127.0.0.1:5000',
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(context, controller.text.trim()),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    if (url != null && url.isNotEmpty) {
      setState(() => AppConfig.baseUrl = url);
      _checkHealth();
    }
  }

  Future<void> _checkHealth() async {
    setState(() => _status = 'Checking backend…');
    try {
      final h = await ApiService.health();
      if (!mounted) return;
      setState(() => _status = 'Connected · model ${h['model']}');
    } catch (_) {
      if (!mounted) return;
      setState(() => _status = 'Backend not reachable at ${AppConfig.baseUrl}');
    }
  }

  @override
  Widget build(BuildContext context) {
    // On desktop/web there's no camera; only offer it where it makes sense.
    final showCamera = !kIsWeb &&
        (defaultTargetPlatform == TargetPlatform.android ||
            defaultTargetPlatform == TargetPlatform.iOS);

    return Scaffold(
      appBar: AppBar(
        title: const Text('ECG Arrhythmia Detector'),
        actions: [
          IconButton(
            tooltip: 'Server settings',
            onPressed: _busy ? null : _editServerUrl,
            icon: const Icon(Icons.dns_outlined),
          ),
        ],
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 520),
          child: ListView(
            padding: const EdgeInsets.all(20),
            shrinkWrap: true,
            children: [
              const _Hero(),
              const SizedBox(height: 28),
              _DropZoneButton(
                onTap: _busy ? null : () => _run(_pickFile),
                onDrop: _handleDrop,
                busy: _busy,
              ),
              const SizedBox(height: 12),
              if (showCamera)
                _ActionButton(
                  icon: Icons.photo_camera_rounded,
                  label: 'Take a photo of an ECG',
                  sub: 'Capture a printout or monitor with the camera',
                  onTap: _busy
                      ? null
                      : () => _run(() => _pickImage(ImageSource.camera)),
                ),
              if (showCamera) const SizedBox(height: 12),
              _ActionButton(
                icon: Icons.image_outlined,
                label: 'Pick an image',
                sub: 'Choose an ECG picture from the gallery',
                onTap: _busy
                    ? null
                    : () => _run(() => _pickImage(ImageSource.gallery)),
              ),
              const SizedBox(height: 24),
              if (_busy)
                const Center(child: CircularProgressIndicator())
              else
                TextButton.icon(
                  onPressed: _checkHealth,
                  icon: const Icon(Icons.wifi_tethering, size: 18),
                  label: const Text('Test backend connection'),
                ),
              if (_status.isNotEmpty) ...[
                const SizedBox(height: 12),
                Text(_status,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                        color: AppColors.muted, fontSize: 12.5)),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _Hero extends StatelessWidget {
  const _Hero();

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: AppColors.surface,
            shape: BoxShape.circle,
            border: Border.all(color: AppColors.border),
          ),
          child: const Icon(Icons.monitor_heart_rounded,
              size: 48, color: AppColors.accent),
        ),
        const SizedBox(height: 18),
        const Text(
          'Arrhythmia analysis',
          style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w800,
              color: AppColors.text),
        ),
        const SizedBox(height: 8),
        const Text(
          'Upload a 12-lead ECG recording or a picture, and the model will '
          'classify the rhythm, show every class probability, and highlight '
          'what it focused on.',
          textAlign: TextAlign.center,
          style: TextStyle(color: AppColors.muted, height: 1.5, fontSize: 13.5),
        ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final String sub;
  final VoidCallback? onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.sub,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppColors.surface,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0x2638BDF8),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: AppColors.accent),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(label,
                        style: const TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 15,
                            color: AppColors.text)),
                    const SizedBox(height: 2),
                    Text(sub,
                        style: const TextStyle(
                            color: AppColors.muted, fontSize: 12)),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: AppColors.muted),
            ],
          ),
        ),
      ),
    );
  }
}

/// Simple container for a picked file's bytes + name.
class _PickedFile {
  final Uint8List bytes;
  final String name;
  _PickedFile(this.bytes, this.name);
}

class _DropZoneButton extends StatefulWidget {
  final VoidCallback? onTap;
  final ValueChanged<DropDoneDetails>? onDrop;
  final bool busy;

  const _DropZoneButton({this.onTap, this.onDrop, this.busy = false});

  @override
  State<_DropZoneButton> createState() => _DropZoneButtonState();
}

class _DropZoneButtonState extends State<_DropZoneButton> {
  bool _isDragging = false;

  @override
  Widget build(BuildContext context) {
    return DropTarget(
      onDragEntered: (_) => setState(() => _isDragging = true),
      onDragExited: (_) => setState(() => _isDragging = false),
      onDragDone: (details) {
        setState(() => _isDragging = false);
        if (widget.onDrop != null && !widget.busy) {
          widget.onDrop!(details);
        }
      },
      child: Material(
        color: _isDragging ? AppColors.surface.withOpacity(0.5) : AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        child: InkWell(
          borderRadius: BorderRadius.circular(14),
          onTap: widget.onTap,
          child: Container(
            padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 16),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: _isDragging ? AppColors.accent : AppColors.border,
                width: _isDragging ? 2 : 1,
              ),
              color: _isDragging ? AppColors.accent.withOpacity(0.1) : null,
            ),
            child: Column(
              children: [
                Icon(
                  Icons.cloud_upload_outlined,
                  size: 42,
                  color: _isDragging ? AppColors.accent : AppColors.muted,
                ),
                const SizedBox(height: 12),
                Text(
                  _isDragging ? 'Drop file here' : 'Click to browse or drop a file here',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                    color: _isDragging ? AppColors.accent : AppColors.text,
                  ),
                ),
                const SizedBox(height: 6),
                const Text(
                  '.pt · .mat · .dat · image · PDF',
                  style: TextStyle(color: AppColors.muted, fontSize: 12),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

