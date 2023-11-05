import pathlib
from tqdm import tqdm
import sys

from env_file import *
import shutil

assert len(sys.argv) == 3, 'Invalids args, i.g: python <root_data_dir> <new_data_dir>'

# ROOT_DS_DIR =  '../data'
# NEW_DS_DIR = '../x-ray-all-exclude-val-4/'
ROOT_DS_DIR = sys.argv[0]
NEW_DS_DIR = sys.argv[2]

new_ds_dir= pathlib.Path(NEW_DS_DIR)
new_ds_dir.mkdir(exist_ok=True)
new_image_train_dir = new_ds_dir / 'images' / 'train'
new_image_train_dir.mkdir(exist_ok=True)
new_label_train_dir = new_ds_dir / 'labels' / 'train'
new_label_train_dir.mkdir(exist_ok=True)

root_ds_dir = pathlib.Path(ROOT_DS_DIR)
ds_image_dirs = root_ds_dir.glob(f'*/{IMAGES_DIRECTORY}/')

for ds_image_dir in tqdm( ds_image_dirs ):
    ds_name = ds_image_dir.parent.name
    ds_images = ds_image_dir.iterdir()
    for image_path in tqdm( ds_images, leave=False ):
        image_name = image_path.name
        dst_file_path = new_image_train_dir  / f"{ds_name}_{image_name}"
        shutil.copy(image_path, dst_file_path)

        label_file = LABEL_BBOX_DIRECTORY.join( str(image_path).rsplit( IMAGES_DIRECTORY, 1) ).rsplit('.', 1)[0]+'.txt'
        label_path = pathlib.Path(label_file)
        if not label_path.is_file():
            print(f"{label_path} is not existed!")
            continue
        label_name = label_path.name
        dst_label_path = new_label_train_dir / f"{ds_name}_{label_name}"
        shutil.copy(label_path, dst_label_path)

