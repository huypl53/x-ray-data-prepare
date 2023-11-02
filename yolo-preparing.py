from os.path import isdir, split
import sys
import os
from glob import glob
from env_file import *
from tqdm import tqdm
import random
from typing import List
import shutil

train_test_valid_rates = [0.7, 0., 0.3]
assert len( train_test_valid_rates ) == 3 and sum(train_test_valid_rates) == 1

def split_data(data_list: List[str], rate_list: List[float]) -> List[List[str]]:
    data_len = len(data_list)
    rate_list = [int( data_len*v ) for v in rate_list]
    curr_ind = 0
    split_data = []
    for v in rate_list:
        split_data.append(data_list[curr_ind: curr_ind + v])
        curr_ind += v
    return split_data


_root_dataset_dir = 'datasets'
if __name__ == '__main__':
    root_dir = ''
    os.makedirs(_root_dataset_dir,exist_ok=True)
    try:
        root_dir = sys.argv[1]
        if not os.path.isdir(root_dir):
            raise Exception
    except: 
        print('Root dir must be specified and valid...')
        exit()
    image_dirs = glob(f'{root_dir}/**/{IMAGES_DIRECTORY}', recursive=True)
    image_dirs = [d for d in image_dirs if YOLO_DIR not in d]
    for image_dir in tqdm( image_dirs ):
        sub_root_dir = os.path.dirname(image_dir)
        dataset_name = os.path.basename(sub_root_dir)
        yolo_root_dir = os.path.join(sub_root_dir, YOLO_DIR)
        os.makedirs(yolo_root_dir, exist_ok=True)

        yolo_image_dir = os.path.join(yolo_root_dir, 'images')
        yolo_label_dir = os.path.join(yolo_root_dir, 'labels')
        if os.path.isdir(yolo_image_dir):
            shutil.rmtree(yolo_image_dir)

        if os.path.isdir(yolo_label_dir):
            shutil.rmtree(yolo_label_dir)

        os.makedirs(yolo_image_dir,exist_ok=True)
        os.makedirs(yolo_label_dir,exist_ok=True)

        image_paths = glob(f'{image_dir}/*')
        random.shuffle(image_paths)
        train, test, val = split_data(image_paths, train_test_valid_rates)
        dataset = {
            'train': train,
            'test': test,
            'val': val
        }

        for k, v in tqdm( dataset.items(), leave=False ):
            images_dir = os.path.join(yolo_image_dir,k)
            labels_dir = os.path.join(yolo_label_dir,k)

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            for image_path in tqdm( v, leave=False ):
                label_base  = os.path.splitext( LABEL_BBOX_DIRECTORY.join(image_path.rsplit(IMAGES_DIRECTORY, 1)))[0]
                label_path = label_base +'.txt'

                shutil.copy2(image_path, images_dir)
                shutil.copy2(label_path, labels_dir)
        open(os.path.join( f"{_root_dataset_dir}", f"{dataset_name}-coco.yaml" ), 'w', encoding='utf-8' ).write(
f"""
path: {yolo_root_dir}  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes (80 COCO classes)
names:
  0: lesion
"""
        )

