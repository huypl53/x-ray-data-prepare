import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from pandas import DataFrame
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO, settings
from ultralytics.cfg import get_cfg  # , get_save_dir

# from sklearn.metrics import precision_score, accuracy_score, multilabel_confusion_matrix
from ultralytics.utils import SETTINGS, ops
from ultralytics.utils.metrics import ap_per_class

from utils import box_iou, read_label, rel2abs

MAX_IM_NUM = -1
EPS = 1e-8

confidences = [0.3, 0.5, 0.7, 0.9]

# metrics based on https://en.wikipedia.org/wiki/Precision_and_recall#Definition
def get_save_dir(args):
    project = args.project or Path(SETTINGS["runs_dir"]) / args.task
    name = args.name or f"{args.mode}"
    return Path(project) / name


def calculate_metrics(im_metric: Dict[str, float]) -> List[float]:
    im_tp = im_metric["im_tp"]
    im_tn = im_metric["im_tn"]
    im_fp = im_metric["im_fp"]
    im_fn = im_metric["im_fn"]

    # Metics on images
    im_se = im_tp / (im_tp + im_fn + EPS)
    im_sp = im_tn / (im_fp + im_tn + EPS)
    im_ppv = im_tp / (im_tp + im_fp + EPS)
    im_npv = im_tn / (im_fn + im_tn + EPS)
    im_acc = (im_tp + im_tn) / (im_tp + im_fp + im_fn + im_tn + EPS)

    # Metrics on classes
    matrix = Matrix(cls_matrix[conf].shape[0])
    matrix.matrix = cls_matrix[conf]
    cls_tp = matrix.tp[0]
    cls_fp = matrix.fp[0]
    cls_tn = matrix.tn[0]
    cls_fn = matrix.fn[0]
    cls_se = cls_tp / (cls_tp + cls_fn + EPS)
    cls_ppv = cls_tp / (cls_tp + cls_fp + EPS)
    cls_f1 = 2 * cls_se * cls_ppv / (cls_se + cls_ppv + EPS)

    return [
        im_se,
        im_sp,
        im_ppv,
        im_npv,
        im_acc,
        cls_se,
        cls_ppv,
        cls_f1,
    ]


def filter_bboxes_wh_ratio(
    bboxes: torch.Tensor, wh_ratios: float = 0.1
) -> torch.Tensor:
    """
    Args:
        bboxes(np.ndarray): N*xywh[cls, score,...]
        wh_ratios(float) < 1
    Returns:
        np.ndarray: same shape as input bboxes
    """
    assert wh_ratios < 1
    filter_preds = bboxes[bboxes[..., 2] / bboxes[..., 3] > wh_ratios]
    filter_preds = filter_preds[filter_preds[..., 3] / filter_preds[..., 2] > wh_ratios]

    return filter_preds


def post_process_bboxes(
    detections: torch.Tensor, wh_ratios: float | None = None, iou: float | None = None
) -> torch.Tensor:

    preds_xywh = ops.xyxy2xywh(detections[..., :4])
    preds_xywh = torch.cat((preds_xywh, detections[..., 4:]), -1)

    filter_detections = preds_xywh
    # filter by w/h rate
    if wh_ratios:
        filter_detections = filter_bboxes_wh_ratio(filter_detections, wh_ratios)
    #
    filter_detections = torch.cat(
        (ops.xywh2xyxy(filter_detections[..., :4]), filter_detections[..., 4:]),
        -1,
    )
    if iou:
        # filter by overlap rate
        filter_detections = filter_bboxes_overlap(filter_detections, iou)
    return filter_detections


class Matrix:
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if size is int:
            self.matrix = np.zeros((size, size))

        else:
            self.matrix = np.zeros(size)

    @property
    def tp(self):
        return self.matrix.diagonal()

    @property
    def fp(self):
        return self.matrix.sum(1) - self.tp

    @property
    def fn(self):
        return self.matrix.sum(0) - self.tp

    @property
    def tn(self):
        result = np.zeros(self.matrix.shape[0])
        for i in range(result.shape[0]):
            mask = np.ones_like(self.matrix)
            mask[i, :] = 0
            mask[:, i] = 0
            result[i] = np.sum(self.matrix * mask)

        return result


class ConfidentMatrix(Matrix):
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, single_class=True, task="detect"):
        """Initialize attributes for the YOLO model."""
        super().__init__((nc + 1, nc + 1) if task == "detect" else (nc, nc))
        self.task = task
        self.nc = nc  # number of classes
        self.conf = (
            0.25 if conf in {None, 0.001} else conf
        )  # apply 0.25 if default val conf is passed
        self.iou_thres = iou_thres
        self._single_class = single_class

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (torch.Tensor[N, 6] | torch.Tensor[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            gt_bboxes (torch.Tensor[M, 4]| torch.Tensor[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_cls (torch.Tensor[M]): The class labels.
        """
        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
            return
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        # is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # with additional `angle` dimension
        is_obb = False
        iou = (
            # batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            # if is_obb
            # else box_iou(gt_bboxes, detections[:, :4])
            box_iou(gt_bboxes, detections[:, :4])
        )

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

        if self._single_class:
            matrix = np.zeros((2, 2))
            matrix[0, 0] = np.sum(self.matrix[:-1, :-1])
            matrix[0, 1] = np.sum(self.matrix[:-1, -1])
            matrix[1, 0] = np.sum(self.matrix[-1, :-1])
            matrix[1, 1] = self.matrix[-1, -1]

            self.matrix = matrix


def filter_bboxes_overlap(bboxes: torch.Tensor, overlap=0.7) -> torch.Tensor:
    """
        Args:
        bboxes(torch.Tensor): N*xyxy[cls, score,...]
    Returns:
        torch.Tensor: same shape as input bboxes
    """
    if bboxes.shape[0] == 0:
        return bboxes
    bboxes_xyxy = bboxes[..., :4]
    iou_matrix = box_iou(bboxes_xyxy, bboxes_xyxy)
    iou_matrix[iou_matrix == 1] = 0
    overlap_idx = torch.max(torch.triu(iou_matrix), 1).values < overlap

    return bboxes[overlap_idx]


def gen_prediction(model, im_paths):
    for im_path in im_paths:
        yield model(im_path)


if __name__ == "__main__":

    im_dir = sys.argv[1]
    lb_dir = sys.argv[2]
    data_config = sys.argv[3]
    model_path = sys.argv[4]
    task = sys.argv[5]
    # save_suffix = ""

    assert task in ["detect", "segment"]

    # if len(sys.argv) > 4:
    #     save_suffix = sys.argv[4]
    model = YOLO(model_path)  # pretrained YOLOv7n model

    im_paths = glob(f"{im_dir}/*")[:MAX_IM_NUM]
    lb_paths = [
        os.path.join(lb_dir, os.path.basename(im_path).split(".")[0] + ".txt")
        for im_path in im_paths
    ]

    results_gen = gen_prediction(model, im_paths)

    results = dict()

    args = dict(model=model_path, data=data_config, conf=0.5, mode="val", task=task)
    args = get_cfg(overrides=args)
    save_dir = Path(f"{str(get_save_dir(args))}")
    nc = len(model.names)  # num class

    im_metrics: Dict[float, Dict[str, float]] = dict()

    cls_matrix: Dict[float, np.ndarray] = dict()
    conf_cfg = {
        "nc": nc,
        "conf": 0.5,
        "iou_thres": 0.45,
        "single_class": True,
        "task": "detect",
    }
    for conf in confidences:
        im_metrics[conf] = {
            "im_tp": 0.0,
            "im_fp": 0.0,
            "im_tn": 0.0,
            "im_fn": 0.0,
        }
        cls_matrix[conf] = np.zeros((2, 2))

    for i, (im_path, lb_path, result) in tqdm(
        enumerate(zip(im_paths, lb_paths, results_gen))
    ):

        image = Image.open(im_path)
        draw_image = ImageDraw.Draw(image)
        for conf in confidences:
            conf_args = {**conf_cfg, "conf": conf}
            r = result[0]
            gt_bboxes = read_label(lb_path)
            im_h, im_w = r.orig_shape[:2]
            bbox_xywh = rel2abs(gt_bboxes, im_w, im_h)
            gt_xyxy = torch.stack(
                [ops.xywh2xyxy(torch.tensor(d[1])) for d in bbox_xywh]
            )
            gt_cls = torch.tensor([d[0] for d in bbox_xywh])

            detections = (
                r.boxes.data.cpu().detach()
            )  # [ [ x1, y1, x2, y2, cnf, cls ] ] # abs coords

            filter_detections = post_process_bboxes(detections, 0.2, 0.7)
            conf_matrix = ConfidentMatrix(**conf_args)
            conf_matrix.process_batch(filter_detections, gt_xyxy, gt_cls)

            # 1. update overall matix
            cls_matrix[conf] += conf_matrix.matrix

            tp = conf_matrix.tp
            fp = conf_matrix.fp
            tn = conf_matrix.tn
            fn = conf_matrix.fn

            if len(bbox_xywh) > 0:  # 2.1 anh duong goc
                # 2.1.1.  anh duong tu anh duong goc
                if tp[0] >= fn[0]:
                    im_metrics[conf]["im_tp"] += 1
                else:
                    im_metrics[conf]["im_fn"] += 1

                # 2.1.2.  anh am tu anh duong goc
                if fp[0] > 0:
                    im_metrics[conf]["im_fp"] += 1
                else:
                    im_metrics[conf]["im_tn"] += 1
            else:  # 2.2. anh am goc
                if fp[0] > 0:
                    im_metrics[conf]["im_fp"] += 1
                else:
                    im_metrics[conf]["im_tn"] += 1

    for conf in confidences:
        im_metric = im_metrics[conf]
        results[conf] = calculate_metrics(im_metric)

    df = DataFrame.from_dict(
        results,
        orient="index",
        columns=[
            "im_se",
            "im_sp",
            "im_ppv",
            "im_npv",
            "im_acc",
            "cls_se",
            "cls_ppv",
            "cls_f1",
        ],
    )
    save_file = str(save_dir) + ".csv"
    df.to_csv(save_file, sep="\t")

    pass
