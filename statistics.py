import sys
import pathlib
import os
from glob import glob
from tqdm import tqdm

from utils import parse_label_file, label_num2str

ROOT_DIR = ''
LABEL_DIR = ''
LABEL_EXT = 'txt'

assert len(sys.argv) == 4

ROOT_DIR = sys.argv[1]
LABEL_DIR = sys.argv[2]
LABEL_EXT = sys.argv[3]

label_files = glob(f'{ROOT_DIR}/**/{LABEL_DIR}/*.{LABEL_EXT}')

MIN_NUM_POINT = 20

with open('./lim-outline-files.txt', 'w') as fw: 

    output_files = []
    for label_file in tqdm(label_files):
        labels = parse_label_file(label_file)
        has_tiny_labels = False
        if len(labels) == 0:
            continue
        for label in labels:
            if (len(label)) < MIN_NUM_POINT + 1 and len(label) > 0:
                basename = os.path.basename(label_file)
                file_name = os.path.splitext(basename)[0]
                output_files.append(file_name)
                break

    fw.write( '\n'.join(output_files))
