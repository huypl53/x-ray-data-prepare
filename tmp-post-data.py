import sys
from tqdm import tqdm
from glob import glob
from utils import parse_label_file, save_labels

def fix_unbound_values(labels):
    for label in labels:
        if not len(label): 
            continue
        label[0] = int(label[0])
        for i in range(1, len(label)):
            v = label[i]
            v = max(0.0, v)
            v = min(1.0, v)
            label[i] = v
    
    return labels


label_pattern = sys.argv[1]

label_files = glob(label_pattern)

for label_file in tqdm( label_files, leave=False ):
    labels = parse_label_file(str( label_file ))
    labels = fix_unbound_values(labels)
    save_labels(labels, label_file)
