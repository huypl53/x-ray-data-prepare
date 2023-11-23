import sys
import pathlib
import os
from glob import glob
from tqdm import tqdm

from utils import parse_label_file, label_num2str

ROOT_DIR = ''
LABEL_DIR = ''
LABEL_EXT = '.txt'

assert len(sys.argv) == 4

ROOT_DIR = sys.argv[1]
LABEL_DIR = sys.argv[2]
LABEL_EXT = sys.argv[3]

label_files = glob(f'{ROOT_DIR}/**/{LABEL_DIR}/*.{LABEL_EXT}')

MIN_NUM_POINT = 20

with open('./lim-outline-files.txt', 'a') as fw: 

    for label_file in tqdm(label_files):
        labels = parse_label_file(label_file)
        if (len(labels)) < MIN_NUM_POINT + 1:
            label_str = label_num2str(labels)
            fw.write(label_str + '\n')
