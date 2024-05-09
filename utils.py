import numpy as np


def parse_label_file(file_path: str):
    label_string = open(file_path).read()
    label_string_lines = label_string.split("\n")

    labels = [[float(v) for v in line.split()] for line in label_string_lines]
    for label in labels:
        if not len(label):
            continue
        label[0] = int(label[0])

    return labels


def label_num2str(labels):
    label_lines = [" ".join([str(v) for v in label]) for label in labels]
    label_str = "\n".join(label_lines)
    return label_str


def save_labels(labels, file_path):
    label_str = label_num2str(labels)
    with open(file_path, "w") as fw:
        fw.write(label_str)


def read_label(lb_path):
    lines = open(lb_path, "r").read().split("\n")
    masks = []
    for line in lines:
        l = line.split()
        if not len(l):
            continue
        mask = []
        mask.append(int(l[0]))
        mask.extend([float(p) for p in l[1:]])

        masks.append(mask)
    return masks


def rel2abs(rel_masks, im_w: int, im_h: int):
    """
    return: [[cls_id, [[x1, y1], [x2, y2], ..., [xn, yn]]]]
    """
    abs_masks = []
    for m in rel_masks:
        abs_mask = []
        abs_mask.append(m[0])

        vertices = np.array(m[1:])
        vertices[0::2] *= im_w
        vertices[1::2] *= im_h

        abs_mask.append(vertices.tolist())

        abs_masks.append(abs_mask)
    return abs_masks
