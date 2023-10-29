from PIL import Image, ImageDraw
from sys import argv
from os.path import isdir, join, splitext, basename, dirname
import numpy as np
import json
import cv2

BBOX_LABEL_DICT = {
    '0': 'Lesion'
}
mask = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
def merge_image_and_mask(image, mask, output_path):

    # Resize the mask to match the size of the image
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Extract the alpha channel from the mask
    alpha = mask_resized[:, :, 3] / 255.0

    # Expand the alpha channel to 3 channels for blending
    alpha = np.expand_dims(alpha, axis=2)

    # Multiply the image and mask with the expanded alpha channel
    merged = alpha * mask_resized[:, :, :3] + (1 - alpha) * image
    merged = merged[0:640, 0:640]
    #print(merged.shape)
    # Save the merged image
    cv2.imwrite(output_path, merged)

def handle_image(image_path,
                 IMAGES_DIRECTORY,
                 METADATA_DIRECTORY, 
                 LABEL_OUTLINE_IMAGES_DIRECTORY, 
                 MASK_IMAGES_DIRECTORY,
                 LABEL_IMAGES_DIRECTORY,
                 LABEL_BBOX_DIRECTORY):
    image = Image.open(image_path)
    src_width, src_height = image.size
    image = image.crop(( 0, 0, 1280, min(image.height, 959)  ))
    image.save(image_path)
    image_size = image.size
    width, height = image_size
    new_relative_width = width/src_width 
    new_relative_height = height/src_height
    
    parent_dir = dirname( dirname(image_path) )
    image_name = basename(image_path)
    if not image_name:
        print(f'{image_name} does not exist!')
        return
    image_id = splitext(image_name)[0] 
    meta_path = join(parent_dir, METADATA_DIRECTORY, image_id + '.json')
    metadata = json.load(
        open(meta_path, encoding='UTF-8'))

    mask_image = Image.new('1', image_size)
    mask_draw = ImageDraw.Draw(mask_image)
    label_image = Image.new('P', image_size, '#000')
    label_draw = ImageDraw.Draw(label_image)
    
    label_outline_image = image.copy()
    label_outline_draw = ImageDraw.Draw(label_outline_image)

    # colors_to_label_names = dict()

    region_list_key, region_key = '', ''
    if 'polypRegions' in metadata.keys():
        # global
        region_list_key = 'polypRegions'
        region_key = 'region'
    elif 'region_list' in metadata.keys():
        region_list_key = 'region_list'
        region_key = 'border'
    else:
        print(f'No key metadata label found with image: {image_path}')
        return


    bbox_labels = []
    try:
        for item in metadata[region_list_key]:
            xmin, ymin, xmax, ymax = 1,1,0,0
            vertices = []
            for v in item[region_key]['vertices']:
                v['x'] = v['x']/ new_relative_width
                v['y'] = v['y']/ new_relative_height
                vertices.append((v['x'] * width, v['y'] * height))
                xmin = min(xmin, v['x'])
                xmax = max(xmax, v['x'])
                ymin = min(ymin, v['y'])
                ymax = max(ymax, v['y'])
            mask_draw.polygon(vertices, fill=1, outline=1)
            if (item['label'] is not None):
                item_color = item['label']['color']
                # item_label_name = item['label']['displayName']
                label_draw.polygon(vertices, fill=item_color, outline=item_color)
                label_outline_draw.polygon(vertices, outline=item_color)
                # colors_to_label_names[item_color] = item_label_name

            center_bbox_label = [(xmax + xmin)/2, (ymax + ymin)/2, xmax - xmin, ymax -ymin]
            # TODO: label mapping 
            bbox_label = ' '.join(['0'] + [str(v) for v in center_bbox_label])
            bbox_labels.append(bbox_label)
    except Exception as e:
        print(f"Worker failed at: {meta_path}, {e}" )
        exit()
        # return dict()


    open(join(parent_dir, LABEL_BBOX_DIRECTORY, image_id + '.txt'), 'w').write('\n'.join(bbox_labels))

    label_outline_image.save(join(parent_dir, LABEL_OUTLINE_IMAGES_DIRECTORY, image_id + '.png'))
    mask_image.save(join(parent_dir, MASK_IMAGES_DIRECTORY, image_id + '.png'))
    label_image.save(join(parent_dir, LABEL_IMAGES_DIRECTORY, image_id + '.png'))
    # return colors_to_label_names

def get_image_meta(meta_path):
    try:
        metadata = json.load(
            open(meta_path, encoding='UTF-8'))
        return metadata
    except Exception as e:
        print(f"Read {e}: {meta_path}")
        return dict()


if __name__ == '__main__':
    pass
    # image_id = argv[1]
    # print(handle_image(image_id))
