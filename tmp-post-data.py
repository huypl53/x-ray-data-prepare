import sys
from tqdm import tqdm
from glob import glob

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

def parse_label_file(file_path: str): 
    label_string = open(file_path).read()
    label_string_lines = label_string.split('\n')
    
    labels = [ [float(v) for v in line.split()] for line in label_string_lines]
    for label in labels:
        if not len(label): 
            continue
        label[0] = int(label[0])

    return labels

def save_labels(labels, file_path):
    label_lines = [ ' '.join( [str(v) for v in label] ) for label in labels]
    label_str = '\n'.join(label_lines)
    with open(file_path, 'w') as fw:
        fw.write(label_str)

label_pattern = sys.argv[1]

label_files = glob(label_pattern)

for label_file in tqdm( label_files, leave=False ):
    labels = parse_label_file(str( label_file ))
    labels = fix_unbound_values(labels)
    save_labels(labels, label_file)
