from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
import sys
from ultralytics.cfg import get_cfg #, get_save_dir
from ultralytics.utils.metrics import ap_per_class
from ultralytics import settings
import torch
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, multilabel_confusion_matrix

from ultralytics.utils import  SETTINGS
# from ultralytics.utils import LOGGER, TQDM
# from ultralytics.utils.torch_utils import de_parallel, select_device
# from ultralytics.nn.autobackend import AutoBackend
# from ultralytics.utils.ops import Profile
# from ultralytics.utils.plotting import output_to_target, plot_images


def get_save_dir(args):                                                            
    project = args.project or Path(SETTINGS['runs_dir']) /args.task     
    name =args.name or f'{args.mode}'                                        
    return Path(project) / name
    # return increment_path(Path(project) / name, exist_ok=self.args.exist_ok)   

def confusion_matrix2pred_groundtruth(matrix):
    pr_ind, gt_ind = 0, 0

    max_pre_num = int( np.sum(matrix) )
    preds = np.zeros(max_pre_num)
    gts = np.zeros(max_pre_num)

    for pr_cls, row in enumerate( matrix ):
        for gt_cls, v in enumerate(row):
            v = int(v)
            preds[pr_ind: pr_ind+v] = pr_cls
            pr_ind += v

            gts[gt_ind: gt_ind+v] = gt_cls
            gt_ind += v

    real_gt_num = max(pr_ind, gt_ind)
    preds = preds[: real_gt_num]
    gts = gts[: real_gt_num]

    return preds, gts

# metrics based on https://en.wikipedia.org/wiki/Precision_and_recall#Definition
model_path = sys.argv[1]
data_config = sys.argv[2]
task = sys.argv[3]
save_suffix = ''

assert task in ['detect', 'segment']

if len(sys.argv) > 4:
    save_suffix =  sys.argv[4]

confidences = [0.3, 0.5, 0.7, 0.9]
for conf in confidences:
    save_suffix =  '_' + str(conf)

    args = dict(model=model_path, data=data_config, conf=conf, mode='val', task=task)
    args = get_cfg(overrides=args)
    # print(f"{args}")
    save_dir = get_save_dir(args)
    save_dir = Path(f"{str(save_dir)}_{conf}")
    validator = DetectionValidator(save_dir = save_dir, args=args) if task == 'detect' else SegmentationValidator(save_dir = save_dir, args=args)
    validator()

    matrix = validator.confusion_matrix.matrix # np.array

    epsilon = 1e-16
    # [[ tp, fp], [fn , tn]] = matrix
    # p = tp / (tp + fp + epsilon)
    # r = tp / (tp + fn + epsilon)
    #
    # accuracy = (tp+tn)/(tp+tn+fp+fn + epsilon)
    # precision = tp / (tp+fp + epsilon)
    # sensitivity = tp / (tp + fn + epsilon)
    # specificity = tn/ ( tn + fp + epsilon)

    runs_dir = settings[ 'runs_dir' ]
    model_name = os.path.basename(model_path).replace('.', '-')
    preds, gts = confusion_matrix2pred_groundtruth(matrix)

    mcm = multilabel_confusion_matrix(preds, gts, labels=range(len(matrix)))
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    sensitivity = tp / (tp + fn + epsilon) # len = class num
    specificity = tn/ ( tn + fp + epsilon) # len = class num

    accuracy = accuracy_score(preds, gts)
    precision = precision_score(preds, gts, average=None, zero_division=0.0) # return all classes values

    with open(os.path.join(runs_dir, task, f'val-metics-{model_name}-{save_suffix}.csv'), 'w') as fw:
        result = [str( accuracy )] + [','.join( [ str(s) for s in v ] ) for v in (  precision, sensitivity, specificity )  ]
        result_string = '\n'.join( result )
        fw.write(result_string)

