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
    "0": "-",
    "1": "0",
    "2": "Барьер впереди",
    "3": "Крупный рогатый скот",
    "4": "Осторожность",
    "5": "Велосипедный переезд",
    "6": "Опасное падение",
    "7": "Место еды",
    "8": "Падающие камни",
    "9": "Перевозить",
    "10": "Медпункт",
    "11": "Уступи дорогу",
    "12": "Рог запрещен",
    "13": "Больница",
    "14": "горб",
    "15": "Изгиб заколки для волос слева",
    "16": "Левый обратный изгиб",
    "17": "Левая кривая",
    "18": "Легкое освежение",
    "19": "Дорожные работы",
    "20": "Узкий мост",
    "21": "Впереди узкая дорога",
    "22": "Парковка запрещена",
    "23": "Нет остановки",
    "24": "Нет сквозной дороги",
    "25": "Нет тщательной боковой дороги",
    "26": "Парковка автомобилей",
    "27": "Цикл парковки",
    "28": "Парковка для скутеров и мотоциклов",
    "29": "Парковка Эта сторона",
    "30": "Пешеходный переход",
    "31": "Пешеходам запрещено",
    "32": "Бензонасос- АЗС",
    "33": "Общественный телефон",
    "34": "Место отдыха",
    "35": "Правый изгиб заколки для волос",
    "36": "Правая кривая",
    "37": "Правый обратный изгиб",
    "38": "Ширина дороги впереди",
    "39": "Вокруг",
    "40": "Впереди школа",
    "41": "Скользкая дорога",
    "42": "Ограничение скорости -10-",
    "43": "Ограничение скорости -100-",
    "44": "Ограничение скорости -110-",
    "45": "Ограничение скорости -120-",
    "46": "Ограничение скорости -130-",
    "47": "Ограничение скорости -140-",
    "48": "Ограничение скорости -150-",
    "49": "Ограничение скорости -160-",
    "50": "Ограничение скорости -20-",
    "51": "Ограничение скорости -25-",
    "52": "Ограничение скорости -35-",
    "53": "Ограничение скорости -45-",
    "54": "Ограничение скорости -48-",
    "55": "Ограничение скорости -5-",
    "56": "Ограничение скорости -50-",
    "57": "Ограничение скорости -55-",
    "58": "Ограничение скорости -60-",
    "59": "Ограничение скорости -65-",
    "60": "Ограничение скорости -70-",
    "61": "Ограничение скорости -75-",
    "62": "Ограничение скорости -8-",
    "63": "Ограничение скорости -80-",
    "64": "Ограничение скорости -90-",
    "65": "Ограничение скорости 3",
    "66": "Ограничение скорости 30",
    "67": "Ограничение скорости -15-",
    "68": "Ограничение скорости -40-",
    "69": "Крутой подъем",
    "70": "Крутой спуск",
    "71": "Останавливаться",
    "72": "Прямой запрет, вход запрещен",
    "73": "ходьба"
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