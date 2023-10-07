from PIL import Image, ImageDraw
from sys import argv
from os.path import isdir, join, splitext, basename, dirname
import json

def handle_image(image_path,
                 IMAGES_DIRECTORY,
                 METADATA_DIRECTORY, 
                 LABEL_OUTLINE_IMAGES_DIRECTORY, 
                 MASK_IMAGES_DIRECTORY,
                 LABEL_IMAGES_DIRECTORY):
    image = Image.open(image_path)
    image_size = image.size
    width, height = image_size
    
    parent_dir = dirname( dirname(image_path) )
    image_name = basename(image_path)
    if not image_name:
        print(f'{image_name} does not exist!')
        return []
    image_id = splitext(image_name)[0] 
    metadata = json.load(
        open(join(parent_dir, METADATA_DIRECTORY, image_id + '.json'), encoding='UTF-8'))

    mask_image = Image.new('1', image_size)
    mask_draw = ImageDraw.Draw(mask_image)
    label_image = Image.new('P', image_size, '#000')
    label_draw = ImageDraw.Draw(label_image)
    
    label_outline_image = image.copy()
    label_outline_draw = ImageDraw.Draw(label_outline_image)

    colors_to_label_names = dict()

    region_list_key, region_key = '', ''
    if 'polypRegions' in metadata.keys():
        # global
        region_list_key = 'polypRegions'
        region_key = 'region'
    elif 'region_list' in metadata.keys():
        region_list_key = 'region_list'
        region_key = 'border'
    else:
        print(f'No key label found with image: {image_path}')
        return dict()

    for item in metadata[region_list_key]:
        vertices = []
        for v in item[region_key]['vertices']:
            vertices.append((v['x'] * width, v['y'] * height))
        mask_draw.polygon(vertices, fill=1, outline=1)
        if (item['label'] is not None):
            item_color = item['label']['color']
            item_label_name = item['label']['displayName']
            label_draw.polygon(vertices, fill=item_color, outline=item_color)
            label_outline_draw.polygon(vertices, outline=item_color)
            colors_to_label_names[item_color] = item_label_name

    label_outline_image.save(join(parent_dir, LABEL_OUTLINE_IMAGES_DIRECTORY, image_id + '.png'))
    mask_image.save(join(parent_dir, MASK_IMAGES_DIRECTORY, image_id + '.png'))
    label_image.save(join(parent_dir, LABEL_IMAGES_DIRECTORY, image_id + '.png'))
    return colors_to_label_names

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
