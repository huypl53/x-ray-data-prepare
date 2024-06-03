from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_label_file(file_path: str):
    label_string = open(file_path).read()
    label_string_lines = label_string.split("\n")

    labels = [[float(v) for v in line.split()] for line in label_string_lines]
    for label in labels:
        if not len(label):
            continue
        label[0] = int(label[0])

    return labels


def label_num2str(labels: str):
    label_lines = [" ".join([str(v) for v in label]) for label in labels]
    label_str = "\n".join(label_lines)
    return label_str


def save_labels(labels, file_path: str):
    label_str = label_num2str(labels)
    with open(file_path, "w") as fw:
        fw.write(label_str)


def read_label(lb_path: str):
    """
    Each line in file contains 1 box or 1 segment.
    In each line, first number is cls_id, then [ x, y, w, h ] for box or [x1, y1, x2, y2,...] for segment
    All coordinates are relative

    Return:
        [[cls_id, [x, y, w, h]], ...] for bboxes
        [[cls_id, [x1, y1, x2, y2,....]]] for segments
    """
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


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def draw_prediction(
    image_draw: ImageDraw.ImageDraw,
    boxes: List[Tuple[float, float, float, float]],
    color="blue",
    text: str | None = None,
):
    """
    image_draw:
    boxes ([N, 4]): N boxes with each in format [x1, y1, x2, y2]
    """
    xyxy = boxes
    for coordinates in xyxy:
        image_draw.rectangle(coordinates, outline=color)
        if text:
            image_draw.text(coordinates[:2], text)
