from os import mkdir
import os
from os.path import isdir, join, dirname, splitext
from multiprocessing import Pool, cpu_count
from os import listdir
from worker import handle_image, get_image_meta 
from glob import glob
import sys
import json
from tqdm import tqdm
from pprint import pprint

from env_file import *

def prepare_directories(root_dir):

    if (not isdir(join(root_dir, MASK_IMAGES_DIRECTORY ))):
        mkdir(join(root_dir, MASK_IMAGES_DIRECTORY ))
    if (not isdir(join(root_dir, LABEL_IMAGES_DIRECTORY ))):
        mkdir(join(root_dir, LABEL_IMAGES_DIRECTORY ))
    if (not isdir(join(root_dir, LABEL_OUTLINE_IMAGES_DIRECTORY ))):
        mkdir(join(root_dir, LABEL_OUTLINE_IMAGES_DIRECTORY ))
    if (not isdir(join(root_dir, LABEL_BBOX_DIRECTORY ))):
        mkdir(join(root_dir, LABEL_BBOX_DIRECTORY ))

    if (not isdir(join(root_dir, LABEL_SEG_DIRECTORY ))):
        mkdir(join(root_dir, LABEL_SEG_DIRECTORY ))

def handle_image_wrapper( image_path ):
    return handle_image(image_path, 
                IMAGES_DIRECTORY, 
                METADATA_DIRECTORY, 
                LABEL_OUTLINE_IMAGES_DIRECTORY, 
                # '',
                MASK_IMAGES_DIRECTORY, 
                # '',
                LABEL_IMAGES_DIRECTORY, 
                LABEL_BBOX_DIRECTORY,
                LABEL_SEG_DIRECTORY)
                 
def get_image_meta_wrapper (image_path): 
    meta_path = f'{METADATA_DIRECTORY}'.join(
            image_path.rsplit(f'{IMAGES_DIRECTORY}', 1)
        )
    meta_path = splitext(meta_path)[0] + '.json'
    return get_image_meta(
        meta_path
    )

if __name__ == '__main__':
    root_dir = ''
    try:
        root_dir = sys.argv[1]
        if not os.path.isdir(root_dir):
            raise Exception
    except: 
        print('Root dir must be specified and valid...')
        exit()
    image_dirs = glob(f'{root_dir}/**/{IMAGES_DIRECTORY}', recursive=True)
    image_dirs = [d for d in image_dirs if YOLO_DIR not in d and YOLO_SEG_DIR not in d and YOLO_K_FOLD_DIR not in d]
    print('Scanning list:\n', '\n'.join( image_dirs ))
    for image_dir in tqdm( image_dirs ):
        # print('Scanning in: ', image_dir)
        sub_root_dir = dirname(image_dir)
        prepare_directories(sub_root_dir)

        image_paths = glob(f'{image_dir}/*')


        pool = Pool(cpu_count())
        # colors_to_label_names = dict()

        # for image_path in tqdm( image_paths, leave=False ):
        #     handle_image_wrapper(image_path)

        list( tqdm( pool.imap_unordered(handle_image_wrapper, image_paths), leave=False ))  
        # tqdm( pool.imap_unordered(handle_image_wrapper, image_paths), leave=False )
        # for result in tqdm( pool.imap_unordered(handle_image_wrapper, image_paths), leave=False ):
        #     for item in result:
        #         colors_to_label_names[item] = result[item]
        #
        # with open(join('__colors_2_labels__.json'), 'w') as label_file:
        #     json.dump(colors_to_label_names, label_file)

        # meta_keys = set()

        # for metadata in pool.imap_unordered(get_image_meta_wrapper, image_paths):
        #     if 'polypRegions' not in metadata.keys():
        #         for k in metadata.keys():
        #             meta_keys.add(k)

        # open(join(sub_root_dir, './meta_keys.txt' ), 'w').write('\n'.join(meta_keys))
        # open(join('./meta_keys.txt' ), 'w').write('\n'.join(meta_keys))
