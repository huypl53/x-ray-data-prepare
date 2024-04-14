# - load validator: if val then self.dataloader

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
import sys
from ultralytics.cfg import get_cfg #, get_save_dir
from pathlib import Path
from tqdm import tqdm
from ultralytics.utils.ops import Profile
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import de_parallel, select_device
from ultralytics.utils import LOGGER, TryExcept, ops
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import output_to_target, colors
import torch
from ultralytics.utils import  SETTINGS
import numpy as np
import cv2
import os

import contextlib


def log(str):
    print(str)

def get_save_dir(args):                                                            
    project = args.project or Path(SETTINGS['runs_dir']) /args.task     
    name =args.name or f'{args.mode}'                                        
    return Path(project) / name

model_path = sys.argv[1]
data_config = sys.argv[2]
task = sys.argv[3]
conf=0.5

args = dict(model=model_path, data=data_config, conf=conf, mode='val', task=task, batch=4)
args = get_cfg(overrides=args)
# print(f"{args}")
save_dir = get_save_dir(args)
save_dir = Path(f"{str(save_dir)}_{conf}")
validator = DetectionValidator(save_dir = save_dir, args=args) if task == 'detect' else SegmentationValidator(save_dir = save_dir, args=args)

validator()
dt = (
    # latest version
    # Profile(device=validator.device),
    # Profile(device=validator.device),
    # Profile(device=validator.device),
    # Profile(device=validator.device),

    # work with v8.0.1
    Profile(),
    Profile(),
    Profile(),
    Profile(),

)

model = AutoBackend(
    weights=validator.args.model,
    device=select_device(validator.args.device, validator.args.batch),
    dnn=validator.args.dnn,
    data=validator.args.data,
    fp16=validator.args.half,
)
for batch_i, batch in enumerate(validator.dataloader):
    with dt[0]:
        batch = validator.preprocess(batch)

    # Inference
    with dt[1]:
        preds = model(batch["img"]
                      # , augment=augment
                      )

    # Loss
    # with dt[2]:
    #     if self.training:
    #         self.loss += model.loss(batch, preds)[1]

    # Postprocess
    with dt[3]:
        preds = validator.postprocess(preds)

    images = batch['img'].cpu().detach().float().numpy()
    cls = batch["cls"].cpu().float().numpy()
    batch_idx = batch["batch_idx"].cpu().float().numpy()
    bboxes = batch["bboxes"].cpu().float().numpy()
    batch_masks = batch["masks"].detach().cpu().float().numpy() if 'masks' in batch else None

    # bs, _, h, w = images.shape  # batch size, _, height, width
    bs = images.shape[0]

    for i in range(bs):
        idx = batch_idx == i
        classes = cls[idx] # .astype("int")
        im_path = batch['im_file'][i]

        # im = images[i]
        # proc_im = np.copy( im )
        # proc_im = proc_im.transpose(1,2,0)
        # proc_im = np.ascontiguousarray(proc_im)

        if not os.path.isfile(im_path):
            log(f'{im_path} not exists!')
            continue

        gt_im_draw = cv2.imread(im_path)
        h, w = gt_im_draw.shape[:2]
        # proc_im = proc_im[..., ::-1] # BGR to RGB image
        # Ground truth
        if len(bboxes):
            # conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)

            boxes = bboxes[idx]
            is_obb = boxes.shape[-1] == 5  # xywhr
            boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
            boxes[..., 0::2] *= w  # scale to pixels
            boxes[..., 1::2] *= h

            # if len(boxes):
            #     if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
            #         boxes[..., 0::2] *= w  # scale to pixels
            #         boxes[..., 1::2] *= h
            #     elif scale < 1:  # absolute coords need scale if image scales
            #         boxes[..., :4] *= scale
            # classes = cls[idx] + 25
            boxes = np.concatenate((boxes, np.ones(( len(boxes), 1 )) , classes ), -1)
            masks = None
            if batch_masks is not None and len(batch_masks):
                if idx.shape[0] == batch_masks.shape[0]:  # overlap_masks=False
                    image_masks = batch_masks[idx]
                else:  # overlap_masks=True
                    image_masks = batch_masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                # im = gt_im_draw.copy()
                for j in range(len(image_masks)):
                    color = colors(classes[j])
                    mh, mw = image_masks[j].shape
                    if mh != h or mw != w:
                        mask = image_masks[j].astype(np.uint8)
                        mask = cv2.resize(mask, (w, h))
                        mask = mask.astype(bool)
                    else:
                        mask = image_masks[j].astype(bool)
                    with contextlib.suppress(Exception):
                        gt_im_draw[mask] = (gt_im_draw[mask] * 0.4 + np.array(color) * 0.6)
            # TODO: scale gt masks to fit original image
            # TODO: check gt mask not present
            # if batch_masks is not None and len(batch_masks):
            #     masks = batch_masks[i]
            #     masks = ops.process_mask(masks, pred[:, 6:], pred[:, :4], batch_im.shape[1:], upsample=True)  # HWC
            gt_im_draw = Results(gt_im_draw, path=im_path, names=validator.names, boxes=boxes).plot(img=gt_im_draw)

                                 # , masks=masks


        # TODO: check the prediction to get mask
        # Prediction
        # res = output_to_target(preds[0], max_det=15)
        # if (len(res) == 3):
        #     batch_id, batch_cls_id, batch_xywh = res
        #     batch_conf, batch_xywh  = batch_xywh[..., -1], batch_xywh[..., :-1]
        #
        # else:
        #     batch_id, batch_cls_id, batch_xywh, batch_conf = res
        # 
        # idx = batch_id == i
        # boxes = batch_xywh[idx]
        # conf = np.expand_dims(  batch_conf[idx], -1 )
        # cls_id = np.expand_dims( batch_cls_id[idx], -1)
        # is_obb = boxes.shape[-1] == 5  # xywhr
        # boxes = ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
        # boxes[..., 0::2] *= w  # scale to pixels
        # boxes[..., 1::2] *= h
        # boxes = np.concatenate((boxes, conf , cls_id ), -1)

        # TODO: check if segment
        # batch_masks = torch.cat(validator.plot_masks, dim=0) if len(validator.plot_masks) else torch.tensor( validator.plot_masks)        
        # proc_im = Results(proc_im, path=im_path, names=validator.names, boxes=boxes, masks=batch_masks).plot(img=proc_im)

        batch_im = images[i]
        pr, proto = preds
        orig_img = np.copy( gt_im_draw )
        for j, pred in enumerate(pr):
            if not len(pred):
                masks = None
            else: 
                masks = ops.process_mask(proto[j], pred[:, 6:], pred[:, :4], batch_im.shape[1:], upsample=True)  # HWC
                # masks = ops.process_mask(proto[j], pred[:, 6:], pred[:, :4], orig_img.shape[:2], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(batch_im.shape[1:], pred[:, :4], orig_img.shape[:2])
            orig_img = Results(orig_img, path=im_path, names=validator.names, boxes=pred[:, :6], masks=masks).plot(img=orig_img)

        basename = os.path.basename(im_path)
        cv2.imwrite( str( save_dir / f"{basename}" ), orig_img)



            ##########33

