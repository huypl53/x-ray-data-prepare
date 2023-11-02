import numpy as np
from ultralytics import YOLO
from env_file import *
import json
import os
import sys

results = {}
ksplit=YOLO_K_SPLIT
# Define your additional arguments here
batch = 16
project = 'kfold_demo'
epochs = 200

model = YOLO('yolov8m.pt')

fold_list_file = sys.argv[1]
try:
    fold_list_file = sys.argv[1]
    if not os.path.isfile(fold_list_file):
        raise Exception
except: 
    print('Root dir must be specified and valid...')
    exit()

ds_yamls = open(fold_list_file, 'r').read().split('\n')
for k in range(ksplit):
    dataset_yaml = ds_yamls[k]
    model.train(data=dataset_yaml,epochs=epochs, batch=batch, project=project)  # include any train arguments
    results[k] = model.metrics  # save output metrics for further analysis


