from ultralytics.models.yolo.detect import DetectionValidator
import sys
from ultralytics.utils.metrics import ap_per_class
from ultralytics import settings
import torch
import os

model_path = sys.argv[1]
data_config = sys.argv[2]

args = dict(model=model_path, data=data_config)
validator = DetectionValidator(args=args)
validator()

matrix = validator.confusion_matrix.matrix
epsilon = 1e-16
[[ tp, fp], [fn , tn]] = matrix
p = tp / (tp + fp + epsilon)
r = tp / (tp + fn + epsilon)

accuracy = (tp+tn)/(tp+tn+fp+fn + epsilon)
precision = tp / (tp+fp + epsilon)
sensitivity = tp / (tp + fn + epsilon)
specificity = tn/ ( tn + fp + epsilon)


runs_dir = settings[ 'runs_dir' ]
model_name = os.path.basename(model_path).replace('.', '-')

with open(os.path.join(runs_dir, 'detect', f'val-metics-{model_name}.txt'), 'w') as fw:
    result = [accuracy, precision, sensitivity, specificity]
    result_string = '\t'.join( [str(v) for v in result] )
    fw.write(result_string)

# stats = validator.stats
#
# stats_process = [torch.cat(x, 0).cpu().numpy() for x in zip(*validator.stats)]
# results =  ap_per_class(*stats_process, names=validator.metrics.names)

# ( tp, fp, p, r, *_ ) = results
#
# fn = (tp / r - tp).astype(int)
#
# sensitivity = tp / (tp + fn)
#
# specificity = 
