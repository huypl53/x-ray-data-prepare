from ultralytics.cfg import get_cfg
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils import LOGGER, ops
# TQDM, callbacks, colorstr, emojis
from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.utils.ops import Profile
from ultralytics.engine.validator import BaseValidator
from tqdm import tqdm
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import output_to_target, plot_images
import os

import cv2
import torch


class Predictor(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, task="detect") -> None:
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args = get_cfg(overrides=args)
        self.args.task = task
        self.lb = []
        pass

    def __call__(self, trainer=None, model=None):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        model = AutoBackend(
            model or self.args.model,
            device=select_device(self.args.device, self.args.batch),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
        )
        # self.model = model
        self.device = model.device  # update device
        self.args.half = model.fp16  # update half
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = 1  # export.py models default to batch-size 1
            LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        if str(self.args.data).split(".")[-1] in ("yaml", "yml"):
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == "classify":
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError((f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        if self.device.type in ("cpu", "mps"):
            self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
        if not pt:
            self.args.rect = False
        self.stride = model.stride  # used in get_dataloader() for padding
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        # self.run_callbacks("on_val_start")
        dt = (
            Profile(),
            Profile(),
            Profile(),
            Profile(),
        )
        bar = tqdm(self.dataloader, 
                   # desc=self.get_desc(),
                   total = len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            # self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            # with dt[2]:
            #     if self.training:
            #         self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                ori_images = []
                for im_path in batch['im_file']:
                    im = cv2.imread(im_path)
                    ori_images.append(im)


            label_images = []
            # height, width = batch["img"].shape[2:]
            # bboxes = batch['bboxes'] * torch.tensor((width, height, width, height), device=self.device)
            # bboxes = torch.cat((bboxes, batch['cls'], torch.ones(len(batch['cls']), 1).to(self.device)), 1)

            # classes = cls[idx].astype("int")
            # labels = confs is None
            batch_idx = batch["batch_idx"]
            cls = batch["cls"] #.squeeze(-1)

            for i in range(len(batch['img'])):
                idx = batch_idx == i
                boxes = batch['bboxes'][idx]
                boxes = ops.xywh2xyxy(boxes)
                orig_img = ori_images[i]
                height, width = orig_img.shape[:2]
                # boxes = boxes * torch.tensor((width, height, width, height), device=self.device)
                boxes[..., 0::2] *= width  # scale to pixels
                boxes[..., 1::2] *= height
                classes = cls[idx] + 25

                boxes = torch.cat((boxes, torch.ones( len(boxes), 1).to(self.device) , classes ), 1)
                # img_path = self.batch[0][i]
                img_path = batch['im_file'][i]
                im = Results(orig_img, path=img_path, names=self.names, boxes=boxes).plot()
                label_images.append(im)
                # basename = os.path.basename(batch['im_file'][i])
                # cv2.imwrite( str( self.save_dir / f"{basename}" ), im)

            preds = self.postprocess(preds, batch['img'], label_images, batch['im_file'])


            for i, pred in enumerate( preds ):
                im_array = pred.plot()
                basename = os.path.basename(batch['im_file'][i])
                cv2.imwrite( str( self.save_dir / f"{basename}" ), im_array)

            # if batch_i < 5:
            #     plot_images(
            #         np.array(predicted_images),
            #         # batch["img"],
            #         batch["batch_idx"],
            #         batch["cls"].squeeze(-1),
            #         batch["bboxes"],
            #         paths=batch["im_file"],
            #         fname=self.save_dir / f"val_batch{batch_i}_labels.jpg",
            #         names=self.names,
            #         on_plot=self.on_plot,
            #     )
            # self.update_metrics(preds, batch)
            # if self.args.plots and batch_i < 3:
            #     self.plot_val_samples(batch, batch_i)
            #     self.plot_predictions(batch, preds, batch_i)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")  # is COCO
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = dict()
        for k, v in  model.names.items():
            k = int(k)
            self.names[k] = v
            self.names[k+25] = f'gt - {v}'
        self.nc = len(self.names)
        # self.metrics.names = self.names
        # self.metrics.plot = self.args.plots
        # self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch
    def postprocess(self, preds, img, orig_imgs, img_paths):
        """Apply Non-maximum suppression to prediction outputs."""
        preds =  ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # img_path = self.batch[0][i]
            img_path = img_paths[i]
            results.append(Results(orig_img, path=img_path, names=self.names, boxes=pred))
        return results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='path to data config file')
    parser.add_argument('-m', '--model', type=str, required=True, help='path to model file')
    parser.add_argument('-t', '--task', type=str, required=True, help='detect, segment, classify')
    parser.add_argument('-c', '--conf', type=float, default=0.5, help='confidence score')
    args = parser.parse_args()


    # args = dict(model=model_path, data=data_config, conf=conf, mode='val', task=task)
    # args = get_cfg(overrides=args)

    custom = {"rect": True}  # method defaults
    args = {
        # **self.overrides, 
        **vars(args),
        **custom, 
        # **kwargs, 
        "mode": "val"}  # highest priority args on the right

    predictor = Predictor(args=args)
    predictor()
