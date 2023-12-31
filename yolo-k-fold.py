import sys
import datetime
import shutil
from pathlib import Path
from collections import Counter
from glob import glob

from env_file import *
import yaml
import pandas as pd
from sklearn.model_selection import KFold
import os
from tqdm import tqdm

try:
    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        raise Exception
except: 
    print('Root dir must be specified and valid...')
    exit()
image_dirs = glob(f'{root_dir}/**/{IMAGES_DIRECTORY}', recursive=True)
image_dirs = [d for d in image_dirs if YOLO_DIR not in d and YOLO_SEG_DIR not in d]
image_dirs = [d for d in image_dirs if YOLO_K_FOLD_DIR not in d]
dataset_dirs = [Path(d).parent for d in image_dirs]

for dataset_dir in tqdm(  dataset_dirs ):
    dataset_path = Path(dataset_dir) # replace with 'path/to/dataset' for your custom data
    labels = sorted(dataset_path.rglob(f"*{LABEL_BBOX_DIRECTORY}/*.txt")) # all data in 'labels'

    yaml_file = './data.yaml'  # your data YAML with data directories and names dictionary
    with open(yaml_file, 'r', encoding="utf8") as y:
        classes = yaml.safe_load(y)['names']
    cls_idx = sorted(classes.keys())

    indx = [l.stem for l in labels] # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    for label in tqdm( labels, leave=False ):
        lbl_counter = Counter()

        with open(label,'r') as lf:
            lines = lf.readlines()

        for l in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(l.split(' ')[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0) # replace `nan` values with `0.0`
    ksplit = YOLO_K_SPLIT
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)   # setting random_state for repeatable results

    kfolds = list(kf.split(labels_df))

    folds = [f'split_{n}' for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=indx, columns=folds)

    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1E-7)
        fold_lbl_distrb.loc[f'split_{n}'] = ratio


# create the directories and dataset YAML files for each split.
    supported_extensions = ['.jpg', '.jpeg', '.png']

# Initialize an empty list to store image file paths
    images = []

# Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / 'images').rglob(f"*{ext}")))

# Create the necessary directories and dataset YAML files (unchanged)
    save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-{YOLO_K_FOLD_DIR}')
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f'{split}_dataset.yaml'
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, 'w') as ds_y:
            yaml.safe_dump({
                'path': split_dir.as_posix(),
                'train': 'train',
                'val': 'val',
                'names': classes
            }, ds_y)
    ( save_path / 'ds_yamls.txt' ).write_text('\n'.join([str(f) for f in ds_yamls ]))
# copy images and labels into the respective directory ('train' or 'val') for each split
    for image, label in zip(images, labels):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / 'images'
            lbl_to_path = save_path / split / k_split / 'labels'

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)

## save the records of the K-Fold split and label distribution DataFrames as CSV files for future reference
    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

