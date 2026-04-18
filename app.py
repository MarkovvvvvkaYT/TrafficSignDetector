import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = 'best.pt'
CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.5

# Словарь перевода (дополните под свои классы)
CLASS_NAMES_RU = {
    'stop': 'Стоп',
    'no entry': 'Въезд запрещён',
    'main road': 'Главная дорога',
    'pedestrian crossing': 'Пешеходный переход',
    'speed limit 50': 'Ограничение 50',
    'speed limit 60': 'Ограничение 60',
    'speed limit 70': 'Ограничение 70',
    'speed limit 80': 'Ограничение 80',
    'speed limit 90': 'Ограничение 90',
    'speed limit 100': 'Ограничение 100',
    'give way': 'Уступите дорогу',
    'no parking': 'Парковка запрещена',
    'turn left': 'Поворот налево',
    'turn right': 'Поворот направо',
}

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    # прогрев для ускорения
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    return model

def get_color_by_distance(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    frame_area = frame_shape[0] * frame_shape[1]
    ratio = area / frame_area
    if ratio < 0.02:
        return (0, 0, 255)   # красный (далеко)
    elif ratio < 0.045:
        return (0, 255, 255) # жёлтый
    else:
        return (0, 255, 0)   # зелёный (близко)

class SignTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.class_names_en = self.model.names

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=640, verbose=False)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, classes):
                en_name = self.class_names_en[cls_id].lower()
                ru_name = CLASS_NAMES_RU.get(en_name, en_name)
                if ru_name.lower() in ("0", "ноль"):
                    continue

                x1, y1, x2, y2 = map(int, box)
                color = get_color_by_distance((x1, y1, x2, y2), img.shape)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{ru_name} ({conf:.2f})"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

st.title("🚦 Детектор дорожных знаков")
st.write("Распознавание в реальном времени с веб-камеры")

ctx = webrtc_streamer(
    key="sign-detector",
    video_transformer_factory=SignTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.state.playing:
    st.success("Камера активна. Направьте на дорожные знаки.")
else:
    st.info("Нажмите 'Start' для запуска камеры.")