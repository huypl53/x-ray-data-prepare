import os
import sys
from pathlib import Path
from glob import glob
from tqdm import tqdm
import shutil

from utils import label_num2str, parse_label_file, save_labels

train_test_valid_rates = [0.7, 0., 0.3]

def change_label(labels, new_class_id):
    new_labels = []
    for label in labels:
        label[0] = int(new_class_id)
        new_labels.append(label)
    return new_labels

if __name__ == '__main__':
    input_dirs = sys.argv[1: -1]
    assert len(input_dirs) < 2
    output_dir = sys.argv[-1]


    output_path = Path(output_dir)
    ( output_path / 'bboxes' ).mkdir()
    ( output_path / 'images' ).mkdir()
    ( output_path / 'keypoints' ).mkdir()

    sub_data_names = []
    for i, input_dir in enumerate( tqdm( input_dirs, leave=False ) ):
        bname = os.path.basename(input_dir)
        sub_data_names.append(bname)
        # Boxes
        dir = os.path.join(input_dir, 'bboxes')
        box_files = glob(f'{dir}/*.txt')
        for box_file in tqdm(box_files, leave=False):
            basename = os.path.basename(box_file)
            labels = parse_label_file(box_file)
            labels = change_label(labels, i)
            label_str = label_num2str(labels) 
            save_labels( label_str, os.path.join(str(output_path), 'bboxes'))


        # Keypoints
        dir = os.path.join(input_dir, 'keypoints')
        point_files = glob(f'{dir}/*.txt')
        for point_file in tqdm(point_files, leave=False):
            basename = os.path.basename(point_file)
            labels = parse_label_file(point_file)
            labels = change_label(labels, i)
            label_str = label_num2str(labels) 
            save_labels( label_str, os.path.join(str(output_path), 'keypoints'))

        # Images
        dir = os.path.join(input_dir, 'images')
        image_files = glob(f'{dir}/*.*')
        dest_dir = os.path.join(output_path, 'images')
        for image_file in tqdm(image_files, leave=False):
            shutil.copy(image_file, dest_dir)


    new_ds_name = '_'.join(sub_data_names)
    open(os.path.join( f"{str( output_path )}", f"{new_ds_name}-bbox-coco.yaml" ), 'w', encoding='utf-8' ).write(
f"""
path: {yolo_bbox_root_dir}  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes (80 COCO classes)
names:
0: lesion
"""
    )


    open(os.path.join( f"{str( output_path )}", f"{dataset_name}-seg-coco.yaml" ), 'w', encoding='utf-8' ).write(
f"""
path: {yolo_seg_root_dir}  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes (80 COCO classes)
names:
0: lesion
"""
    )
