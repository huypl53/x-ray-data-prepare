# Experiments

## Env installation

```bash
pip install -r requirementss.txt
```

## Label generating

Generate segment, bounding boxes from meta data

## Data preprocessing

## Detection task

### Data

```bash
python3 label-generating.py <image_root_folder>
```

1. Yolo data split train/test only

```bash
# prepare train/test + dataset.yml
python3 yolo-preparing.py <image_root_folder>
# training like normal
```

2. Yolo data k-fold

```bash
# preparing
python3 yolo-k-fold.py <image_root_folder>

# training
python yolo_train_fold.py <text_file_contain>
```

## Segmentation task

3. Yolo first training with all data, then finetune with subdataset

```bash

```

## Fix error data

```bash
python tmp-post-data.py <path/to/folder/label/*.txt>
```
