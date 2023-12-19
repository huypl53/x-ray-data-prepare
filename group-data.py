import os
import sys
from pathlib import Path
from glob import glob
from tqdm import tqdm
import shutil
from env_file import YOLO_DIR, YOLO_SEG_DIR

from utils import label_num2str, parse_label_file, save_labels

train_test_valid_rates = [0.7, 0., 0.3]
_root_dataset_dir = 'datasets'

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
    task_dirs = [ YOLO_DIR, YOLO_SEG_DIR ]
    ims_lbs = [ 'images', 'labels' ]
    mode_dirs = [ 'train', 'val' ]

    for task_dir in task_dirs:
        for t in ims_lbs:
            for mode_dir in mode_dirs:

                (output_path / task_dir / t / mode_dir).mkdir()
        

        sub_data_names = []
        sub_data_labels = []
        for i, input_dir in enumerate( input_dirs ):
            bname = os.path.basename(input_dir)
            sub_data_names.append(bname)
            sub_data_labels.append(f'{i}: {bname}')

        sub_data_labels_str = '\n'.join(sub_data_labels)
        new_ds_name = '_'.join(sub_data_names)
        open(os.path.join( f"{_root_dataset_dir}", f"{new_ds_name}-{'bbox' if task_dir == YOLO_DIR else 'seg'}-coco.yaml" ), 'w', encoding='utf-8' ).write(
    f"""
    path: {str(output_path/task_dir)}  # dataset root dir
    train: images/train  # train images (relative to 'path') 4 images
    val: images/val  # val images (relative to 'path') 4 images
    test:  # test images (optional)

# Classes (80 COCO classes)
    names:
    {sub_data_labels_str}
    """
        )


    for i, input_dir in enumerate( tqdm( input_dirs, leave=False ) ):
        bname = os.path.basename(input_dir)
        for task_dir in task_dirs:
            for t in ims_lbs:
                for mode_dir in mode_dirs:
                    dir = os.path.join(input_dir, task_dir, t, mode_dir)
                    dest_dir = os.path.join(str(output_path), task_dir, t, mode_dir)
                    if t == 'images':

                        # Images
                        image_files = glob(f'{dir}/*.*')
                        for image_file in tqdm(image_files, leave=False):
                            shutil.copy(image_file, dest_dir)

                    if t == 'labels':
                        label_files = glob(f'{dir}/*.txt')
                        for label_file in tqdm(label_files, leave=False):
                            basename = os.path.basename(label_file)
                            labels = parse_label_file(label_file)
                            labels = change_label(labels, i)
                            label_str = label_num2str(labels) 
                            save_labels( label_str, os.path.join(dest_dir, basename))


