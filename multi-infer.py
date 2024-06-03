import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw
from ultralytics import YOLO

from utils import draw_prediction

video_path = sys.argv[1]

output_path = "_result.".join(video_path.rsplit(".", 1))

if not os.path.isfile:
    exit(f"{video_path} is not exist!")

_ds = [
    "2_Viem_thuc_quan",
    "3_Viem_da_day_HP_am",
    "5_Ung_thu_thuc_quan",
    "6_Ung_thu_da_day",
    "7_Loet_HTT",
]

model_paths = [
    f"/workspace/detection-240401/{d}/runs/detect/train/weights/best.pt" for d in _ds
]

# colors = list(zip(range(0, 255, 255//9), [128]*10, range(255, 0, -255//9)))
colors = [
    "#FF0000",
    "#00FFFF",
    "#0000FF",
    "#00008B",
    "#ADD8E6",
    "#800080",
    "#FFFF00",
    "#00FF00",
    "#FF00FF",
    "#FFC0CB",
    "#FFFFFF",
    "#C0C0C0",
    "#808080",
    "#000000",
    "#FFA500",
    "#A52A2A",
    "#800000",
    "#008000",
    "#808000",
    "#7FFFD4",
]

models = [YOLO(m) for m in model_paths]


def gen_prediction(models, image):
    for model in models:
        yield model(image)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("cannot open video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 3
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 4

fps = int(cap.get(cv2.CAP_PROP_FPS))  # 5
# cv2.CAP_PROP_FRAME_COUNT   # 7
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot read frame, existing...")
        break

    image = Image.fromarray(frame[..., ::-1])
    image_draw = ImageDraw.Draw(image)

    models_results = gen_prediction(models, frame)
    for i, model_results in enumerate(models_results):
        r = model_results[0]  # infer 1 image -> choose first result

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().detach().tolist()

        model = models[i]
        pr_name = list(model.names.values())[0]
        draw_prediction(image_draw, xyxy, colors[i], pr_name)

    out_frame = np.array(image)[..., ::-1]
    # out_frame = cv2.flip(out_frame, 0) # TODO: necessary?
    out.write(out_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
