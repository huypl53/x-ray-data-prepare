# Data preparing

## Env installation

```bash
pip install -r requirementss.txt
```

## Label generating

Generate segment, bounding boxes from meta data

```bash
python3 label_generating.py <image_root_folder>
```

1. Yolo data split train/test only

```bash
# preparing
python3 yolo-preparing.py <image_root_folder>
# training like normal
```

2. Yolo data k-fold

```bash
# preparing
python3 yolo-k-fold.py <image_root_folder>

# training
python yolo_train_fold.p <text_file_contain>
```

3. Yolo first training with all data, then finetune with subdataset

```bash

```

## Data preprocessing
