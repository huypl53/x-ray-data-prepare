from ultralytics.models.yolo.detect import DetectionValidator
import sys
from ultralytics.utils.metrics import ap_per_class
import torch

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
sensitivity = tp / (tp + fn + epsilon)
specificity = tn/ ( tn + fp + epsilon)

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
