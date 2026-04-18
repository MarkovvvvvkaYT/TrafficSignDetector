import cv2
import time
import threading
import asyncio
import json
import os
import uuid
import numpy as np
from ultralytics import YOLO
import edge_tts
import pygame
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont

# ------------------ НАСТРОЙКИ ------------------
MODEL_PATH = 'best.pt'
CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.5
MIN_SPEAK_INTERVAL = 3.0
TRANSLATIONS_FILE = "translations.json"
VOICE = "ru-RU-SvetlanaNeural"
RATE = "+0%"
VOLUME = "+0%"

# НАСТРОЙКИ ЦВЕТА РАМКИ (доля площади кадра)
FAR_RATIO = 0.02    # меньше этого - красный (далеко)
MID_RATIO = 0.045    # от FAR_RATIO до MID_RATIO - жёлтый, больше - зелёный
# ----------------------------------------------

class VoiceAnnouncer:
    def __init__(self):
        pygame.mixer.init()
        self.voice = VOICE
        self.rate = RATE
        self.volume = VOLUME

    def speak(self, text):
        if not text or not text.strip():
            return
        if text.strip().lower() in ("неизвестно", "0", "ноль", "0.0") or text.strip().isdigit():
            return
        thread = threading.Thread(target=self._run_async, args=(text,), daemon=True)
        thread.start()

    def _run_async(self, text):
        asyncio.run(self._speak_async(text))

    async def _speak_async(self, text):
        tmp_file = f"temp_speech_{uuid.uuid4().hex}.mp3"
        try:
            communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate, volume=self.volume)
            await communicate.save(tmp_file)
            pygame.mixer.music.load(tmp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"Ошибка озвучки: {e}")
        finally:
            if os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except:
                    pass

class TrafficSignDetector:
    def __init__(self, model_path, conf_threshold=0.65, iou_threshold=0.5):
        print("Загрузка модели YOLO...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Прогрев модели (один раз, чтобы ускорить первый кадр)
        print("Прогрев модели...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(dummy, verbose=False)

        self.class_names_en = self.model.names
        print(f"Найдено классов: {len(self.class_names_en)}")

        self.class_names_ru = self._load_or_translate()

        # Фильтрация мусорных классов
        self.class_names_ru = {
            k: v for k, v in self.class_names_ru.items()
            if v not in ("0", "ноль", "Ноль", "0.0") and not v.strip().isdigit()
        }
        print(f"После фильтрации осталось классов (только дорожные знаки): {len(self.class_names_ru)}")

        self.announcer = VoiceAnnouncer()
        self.last_spoken = {}

        # Загрузка шрифта (один раз)
        try:
            self.font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
            except:
                self.font = ImageFont.load_default()
                print("Шрифт Arial не найден, русский текст может отображаться некорректно")

    def _load_or_translate(self):
        if os.path.exists(TRANSLATIONS_FILE):
            print("Загрузка сохранённых переводов...")
            with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        else:
            print("Выполнение перевода (требуется интернет)...")
            translator = Translator()
            translations = {}
            for idx, en_name in self.class_names_en.items():
                try:
                    ru_name = translator.translate(en_name, src='en', dest='ru').text
                    print(f"  {en_name} -> {ru_name}")
                    translations[str(idx)] = ru_name
                except Exception as e:
                    print(f"  Ошибка перевода '{en_name}': {e}")
                    translations[str(idx)] = en_name
            with open(TRANSLATIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            return {int(k): v for k, v in translations.items()}

    def get_color_by_distance(self, bbox, frame_shape):
        """Цвет рамки: красный (далеко) -> жёлтый -> зелёный (близко)"""
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        ratio = area / frame_area

        if ratio < FAR_RATIO:
            return (0, 0, 255)      # красный
        elif ratio < MID_RATIO:
            return (0, 255, 255)    # жёлтый
        else:
            return (0, 255, 0)       # зелёный

    def put_text_russian(self, img, text, position, color):
        """Рисует русский текст (PIL, но без создания нового шрифта каждый раз)"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def process_frame(self, frame):
        # Используем стандартное разрешение 640 (быстрее)
        results = self.model(frame, 
                             conf=self.conf_threshold, 
                             iou=self.iou_threshold, 
                             imgsz=640,   # <-- ВЕРНУЛИ 640 ДЛЯ СКОРОСТИ
                             verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, classes):
                if cls_id not in self.class_names_ru:
                    continue
                ru_name = self.class_names_ru[cls_id]
                if not ru_name or ru_name.lower() in ("0", "ноль", "неизвестно"):
                    continue

                x1, y1, x2, y2 = map(int, box)

                now = time.time()
                if cls_id not in self.last_spoken or (now - self.last_spoken[cls_id]) > MIN_SPEAK_INTERVAL:
                    self.announcer.speak(ru_name)
                    self.last_spoken[cls_id] = now

                color = self.get_color_by_distance((x1, y1, x2, y2), frame.shape)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{ru_name} ({conf:.2f})"
                frame = self.put_text_russian(frame, label, (x1, y1 - 25), color)

        return frame

def main():
    detector = TrafficSignDetector(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)
    print("Детектор запущен. Нажмите 'q' для выхода, 's' для сохранения кадра.")
    detector.announcer.speak("Детектор дорожных знаков запускается")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.process_frame(frame)
        cv2.imshow("Детектор дорожных знаков", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("detected.jpg", frame)
            print("Скриншот сохранён")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()