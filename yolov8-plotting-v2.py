import os
import sys
from ultralytics import YOLO
from glob import glob
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm


MAX_IM_NUM = 100


def read_seg_label(lb_path):
    lines = open(lb_path, "r").read().split("\n")
    masks = []
    for line in lines:
        l = line.split()
        if not len(l):
            continue
        mask = []
        mask.append(int(l[0]))
        mask.extend([float(p) for p in l[1:]])

        masks.append(mask)
    return masks


def rel2abs(rel_masks, im_w: int, im_h: int):
    """
    return: [[cls_id, [[x1, y1], [x2, y2], ..., [xn, yn]]]]
    """
    abs_masks = []
    for m in rel_masks:
        abs_mask = []
        abs_mask.append(m[0])

        vertices = np.array(m[1:])
        vertices[0::2] *= im_w
        vertices[1::2] *= im_h

        abs_mask.append(vertices.tolist())

        abs_masks.append(abs_mask)
    return abs_masks


def gen_prediction(model, im_paths):
    for im_path in im_paths:
        yield model(im_path)


im_dir = sys.argv[1]
lb_dir = sys.argv[2]
model_path = sys.argv[3]
save_dir = sys.argv[4]

os.makedirs(save_dir, exist_ok=True)

im_paths = glob(f"{im_dir}/*")[:MAX_IM_NUM]
lb_paths = [
    os.path.join(lb_dir, os.path.basename(im_path).split(".")[0] + ".txt")
    for im_path in im_paths
]

# TODO: check file exists

model = YOLO(model_path)  # pretrained YOLOv7n model
# results = model(im_paths, stream=True)  # return a generator of Results objects

results = gen_prediction(model, im_paths)

for i, (im_path, lb_path, result) in tqdm(enumerate(zip(im_paths, lb_paths, results))):
    im_name = os.path.basename(im_path)
    save_path = os.path.join(save_dir, im_name)

    image = Image.open(im_path)
    label_outline_draw = ImageDraw.Draw(image)
    im_w, im_h = image.width, image.height

    label_color = "red"
    # Draw label
    masks = read_seg_label(lb_path)
    abs_segment = rel2abs(masks, im_w, im_h)  # x, y label in pixels
    vertices = [seg[1] for seg in abs_segment]
    for vert in vertices:
        label_outline_draw.polygon(vert, outline="blue")

    r = result[0]
    # Draw detection predictions
    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().detach().tolist()
    for coordinates in xyxy:
        label_outline_draw.rectangle(coordinates, outline=label_color)

    # Draw segment predictions
    masks = r.masks  # result[0] due to prediction on one image
    if not masks:
        image.save(save_path)
        continue
    segments = [v.tolist() for mask in masks for v in mask.xyn]
    for seg in segments:
        seg = np.array(seg)
        seg[:, 0] *= im_w
        seg[:, 1] *= im_h
        label_outline_draw.polygon(seg.flatten().tolist(), outline=label_color)

    # print(masks)
    # Save image
    image.save(save_path)
