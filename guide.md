# Experiments

- yolov8 8.0.162

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
python yolo-data-all-then-finetune.py <root_data_dir> <target_data_dir> images
```

## Fix error data

### Fix outbounded labels

```bash
python tmp-post-data.py <path/to/folder/label/*.txt>
```

### Stats tiny objects

```bash
python statistics.py <root/data/dir> <labels/dir/> <txt>
save_dir=report-tiny-labels
mkdir -p $save_dir/images
while IFS= read -r line;
do
    ( find <root/data/dir> -type f -name $line.* -exec cp --parents {} $save_dir/images \;);
done < ./lim-outline-files.txt
```

## Custom metrics

> copy files below to training workspace directory

```bash
# for single models
python metrics.py <path/to/model.pt> <path/to/data_config.yaml> <detect/segment> <save_result_suffix?>

# run with multiple model continuously
# ./export-metrics.sh and metrics.py in same location
bash ./export-metrics.sh <detect/segment>
```

## Remove small boxes

```bash
python remove_small_bbox.py <input-dir> <output-dir>
```

## Inference then save result

```bash
python yolov8-inference.py
```

## Add null image to training

```bash
bash add-null-images.sh <path/to/folder/containing/ds> <path/to/folder/containing/null/images>
```
