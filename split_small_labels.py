import os
from glob import glob
import sys
from tqdm import tqdm
from utils import label_num2str, parse_label_file, save_labels



if __name__ == '__main__':
    # Yolov label in .txt files: split one class with multiple size into different classes
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    assert os.path.isdir(in_dir)
    os.makedirs(out_dir, exist_ok=True)


    # key: new class index (should not overlap original indices), value: size threshhold in ascend order
    categories = {
        1: 0.02,
        2: 0.1,
    }

    in_files = glob(f'{in_dir}/*.*')
    for file_path in tqdm( in_files):
        bboxes = parse_label_file(file_path)
        basename = os.path.basename(file_path)
        new_bboxes = []
        try:
            for box in bboxes:
                min_d = min(box[3], box[4])
                # if min_d < 0.05:
                #     continue
                for cls, thres in categories.items():
                    if min_d < thres:
                        box[0] = cls
                        break
                new_bboxes.append(box)
        except Exception as e: 
            print(f'Error {e} at {file_path}')
            new_bboxes = bboxes

        out_path = os.path.join(out_dir, basename)
        save_labels(new_bboxes, out_path)
