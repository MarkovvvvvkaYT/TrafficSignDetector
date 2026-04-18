import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont

# ------------------ НАСТРОЙКИ ------------------
MODEL_PATH = 'best.pt'
CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.5
TRANSLATIONS_FILE = "translations.json"
# ----------------------------------------------

# Загрузка модели и переводов (кешируем, чтобы не перезагружать при каждом кадре)
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    # Прогрев модели
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    return model

@st.cache_resource
def load_translations():
    if os.path.exists(TRANSLATIONS_FILE):
        with open(TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
    else:
        # Если файла нет, переводим классы (требуется интернет)
        model = YOLO(MODEL_PATH)
        translator = Translator()
        translations = {}
        for idx, en_name in model.names.items():
            try:
                ru_name = translator.translate(en_name, src='en', dest='ru').text
                translations[idx] = ru_name
            except:
                translations[idx] = en_name
        # Сохраняем для будущих запусков
        with open(TRANSLATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in translations.items()}, f, ensure_ascii=False, indent=2)
        return translations

# Загрузка шрифта
@st.cache_resource
def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except:
            return ImageFont.load_default()

def put_text_russian(img, text, position, color, font):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_color_by_distance(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    frame_area = frame_shape[0] * frame_shape[1]
    ratio = area / frame_area
    if ratio < 0.02:
        return (0, 0, 255)   # красный
    elif ratio < 0.045:
        return (0, 255, 255) # жёлтый
    else:
        return (0, 255, 0)   # зелёный

class SignTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.class_names_ru = load_translations()
        self.font = load_font()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=640, verbose=False)

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
                color = get_color_by_distance((x1, y1, x2, y2), img.shape)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{ru_name} ({conf:.2f})"
                img = put_text_russian(img, label, (x1, y1 - 25), color, self.font)

        return img

# Интерфейс Streamlit
st.title("🚦 Детектор дорожных знаков")
st.write("Распознавание знаков в реальном времени с веб-камеры")

webrtc_ctx = webrtc_streamer(
    key="sign-detector",
    video_transformer_factory=SignTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("Камера активна. Направьте на дорожные знаки.")
else:
    st.info("Нажмите 'Start' для запуска камеры.")