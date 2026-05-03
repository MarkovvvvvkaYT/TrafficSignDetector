import 'dart:typed_data';
import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_tts/flutter_tts.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  runApp(const MaterialApp(home: SignDetector(), debugShowCheckedModeBanner: false));
}

class SignDetector extends StatefulWidget {
  const SignDetector({super.key});
  @override
  State<SignDetector> createState() => _SignDetectorState();
}

class _SignDetectorState extends State<SignDetector> {
  CameraController? cameraController;
  Interpreter? interpreter;
  FlutterTts flutterTts = FlutterTts();
  bool isWorking = false;
  bool isInitialized = false;
  List<dynamic> recognitions = [];
  final Map<String, DateTime> _spokenHistory = {};

  final double minConf = 0.40; 
  final double iouThreshold = 0.45;
  
  final Map<int, String> labels = {
    0: "-", 1: "0", 2: "Барьер", 3: "Скот", 4: "Внимание", 6: "Спуск", 
    8: "Камни", 9: "", 10: "", 11: "Уступи дорогу", 12: "Сигнал", 
    13: "", 14: "Неровность", 15: "Поворот налево", 19: "Работы", 
    20: "Мост", 21: "", 22: "Стоянка запрещена", 23: "Остановка запрещена", 
    24: "Тупик", 26: "Автопарковка", 30: "Переход", 31: "Пешеход", 32: "", 
    35: "Поворот направо", 39: "Кольцо", 40: "Дети", 41: "Скользко", 
    42: "Ограничение скорости 10", 43: "Ограничение скорости 100", 50: "Ограничение скорости 20", 56: "Ограничение скорости 50", 
    58: "Ограничение скорости 60", 60: "Ограничение скорости 70", 63: "Ограничение скорости 80", 64: "Ограничение скорости 90", 
    66: "Ограничение скорости 30", 68: "Ограничение скорости 40", 71: "СТОП", 72: "", 73: "Пешеход"
  };

  @override
  void initState() {
    super.initState();
    initApp();
  }

  Future<void> initApp() async {
    await flutterTts.setLanguage("ru-RU");
    await flutterTts.setSpeechRate(0.5);
    if (await Permission.camera.request().isGranted) {
      interpreter = await Interpreter.fromAsset("assets/best_float32.tflite");
      final cameras = await availableCameras();
      cameraController = CameraController(
        cameras[0], 
        ResolutionPreset.medium, 
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      
      await cameraController!.initialize();
      await cameraController!.setExposureMode(ExposureMode.auto);
      
      if (!mounted) return;
      setState(() => isInitialized = true);

      cameraController!.startImageStream((img) {
        if (!isWorking) {
          isWorking = true;
          runModel(img);
        }
      });
    }
  }

  Future<void> runModel(CameraImage img) async {
    if (interpreter == null) return;
    try {
      final shape = interpreter!.getInputTensor(0).shape;
      int mH = shape[1]; 
      int mW = shape[2];
      var input = Float32List(1 * mH * mW * 3);
      
      double zoom = 0.6; 
      int cSize = (min(img.width, img.height) * zoom).toInt();
      int offX = (img.width - cSize) ~/ 2;
      int offY = (img.height - cSize) ~/ 2;

      final p0 = img.planes[0];
      final p1 = img.planes[1];
      final p2 = img.planes[2];

      int p = 0;
      for (int y = 0; y < mH; y++) {
        for (int x = 0; x < mW; x++) {
          int py = offY + (y * cSize ~/ mH);
          int px = offX + (x * cSize ~/ mW);
          
          int yIdx = py * p0.bytesPerRow + px;
          int uvIdx = (py ~/ 2) * p1.bytesPerRow + (px ~/ 2) * p1.bytesPerPixel!;

          int rY = p0.bytes[yIdx];
          int rU = p1.bytes[uvIdx] - 128;
          int rV = p2.bytes[uvIdx] - 128;

          input[p++] = (rY + 1.3707 * rV).clamp(0, 255) / 255.0;
          input[p++] = (rY - 0.3376 * rU - 0.6980 * rV).clamp(0, 255) / 255.0;
          input[p++] = (rY + 1.7324 * rU).clamp(0, 255) / 255.0;
        }
      }

      var outS = interpreter!.getOutputTensor(0).shape;
      var output = List.filled(outS.reduce((a, b) => a * b), 0.0).reshape(outS);
      interpreter!.run(input.reshape(shape), output);

      List<dynamic> results = [];
      for (int i = 0; i < outS[2]; i++) {
        double score = 0; int cls = -1;
        for (int j = 0; j < (outS[1] - 4); j++) {
          if (output[0][j + 4][i] > score) { score = output[0][j + 4][i]; cls = j; }
        }

        String label = labels[cls] ?? "";
        if (score > minConf && label.length > 1 && label != "0") {
          results.add({
            "x": output[0][0][i] / mW, "y": output[0][1][i] / mH,
            "w": output[0][2][i] / mW, "h": output[0][3][i] / mH,
            "label": label, "score": score
          });
        }
      }

      results.sort((a, b) => b["score"].compareTo(a["score"]));
      List<dynamic> finalDet = [];
      for (var b in results) {
        bool keep = true;
        for (var f in finalDet) {
          double xA = max(b["x"]-b["w"]/2, f["x"]-f["w"]/2);
          double yA = max(b["y"]-b["h"]/2, f["y"]-f["h"]/2);
          double xB = min(b["x"]+b["w"]/2, f["x"]+f["w"]/2);
          double yB = min(b["y"]+b["h"]/2, f["y"]+f["h"]/2);
          double inter = max(0.0, xB-xA) * max(0.0, yB-yA);
          if (inter / (b["w"]*b["h"] + f["w"]*f["h"] - inter) > iouThreshold) { keep = false; break; }
        }
        if (keep) finalDet.add(b);
      }

      if (mounted) {
        setState(() => recognitions = finalDet);
        _voiceService(finalDet);
      }
    } catch (e) { print(e); }
    isWorking = false;
  }

  void _voiceService(List<dynamic> detections) {
    final now = DateTime.now();
    for (var d in detections) {
      String l = d["label"];
      if (d["score"] > 0.55 && (_spokenHistory[l] == null || now.difference(_spokenHistory[l]!).inSeconds >= 4)) {
        flutterTts.speak(l);
        _spokenHistory[l] = now;
        break; 
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (cameraController == null || !cameraController!.value.isInitialized) {
      return const Scaffold(backgroundColor: Colors.black, body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: AspectRatio(
          aspectRatio: cameraController!.value.aspectRatio,
          child: Stack(
            children: [
              CameraPreview(cameraController!),
              Positioned(
                top: 30, left: 30,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: recognitions.map((r) => Container(
                    margin: const EdgeInsets.only(bottom: 5),
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(color: Colors.black87, borderRadius: BorderRadius.circular(5)),
                    child: Text("${r['label']} ${(r['score']*100).toInt()}%",
                      style: const TextStyle(color: Colors.greenAccent, fontSize: 18, fontWeight: FontWeight.bold)),
                  )).toList(),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}