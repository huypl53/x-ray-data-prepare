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
# python metrics.py <path/to/model.pt> <path/to/data_config.yaml> <detect/segment> <save_result_suffix?>

declare -a ds=( "2_Viem_thuc_quan" "3_Viem_da_day_HP_am" "5_Ung_thu_thuc_quan" "6_Ung_thu_da_day" "7_Loet_HTT" ); \
for d in "${ds[@]}"; do \

yolo settings datasets_dir=../data/ weights_dir=./$d/weights runs_dir=./$d/runs && python metrics.py  /workspace/data/data-240331/$d/yolov8/images/val/ /workspace/data/data-240331/$d/yolov8/labels/val/ /workspace/detection-240401/$d-bbox-coco.yaml /workspace/detection-240401/$d/runs/detect/train/weights/best.pt  detect;

done

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
# declare -a ds=( "2_Viem_thuc_quan" "3_Viem_da_day_HP_am" "4_Viem_da_day_HP_duong" "5_Ung_thu_thuc_quan" "6_Ung_thu_da_day" "7_Loet_HTT" ); for d in "${ds[@]}"; do \yolo settings datasets_dir=../data/ weights_dir=./$d/weights runs_dir=./$d/runs && python yolov8-inference.py -d datasets/$d-bbox-coco.yaml -m $d/runs/detect/train/weights/best.pt -t detect; done

```

## Add null image to training

```bash
bash add-null-images.sh <path/to/folder/containing/ds> <path/to/folder/containing/null/images>
```

## Plotting results

```bash
declare -a ds=( "2_Viem_thuc_quan" "3_Viem_da_day_HP_am" "4_Viem_da_day_HP_duong" "5_Ung_thu_thuc_quan" "6_Ung_thu_da_day" "7_Loet_HTT" ); \
for d in "${ds[@]}"; do \
    yolo settings datasets_dir=../data/ weights_dir=./$d/weights runs_dir=./$d/runs && python yolov8-plotting.py /workspace/segment-240401/2_Viem_thuc_quan/runs/segment/train/weights/best.pt datasets/2_Viem_thuc_quan-seg-coco.yaml segment; done

# Or
# yolo yolov8-plotting-v2.py <path/yolov8/train/images> <path/yolov8/train/label> <yolov8/model.pt> <path/to/save/dir>

declare -a ds=( "2_Viem_thuc_quan" "3_Viem_da_day_HP_am" "4_Viem_da_day_HP_duong" "5_Ung_thu_thuc_quan" "6_Ung_thu_da_day" "7_Loet_HTT" ); \
for d in "${ds[@]}"; do \

 python yolov8-plotting-v2.py /workspace/data/data-240331/$d/yolov8/images/val/ /workspace/data/data-240331/$d/yolov8/labels/val/ ./$d/runs/detect/train/weights/best.pt ./$d/runs/detect/val/images/ detect; done
 python yolov8-plotting-v2.py /workspace/data/data-240331/$d/yolov8_seg/images/val/ /workspace/data/data-240331/$d/yolov8_seg/labels/val/ ./$d/runs/segment/train/weights/best.pt ./$d/runs/segment/val/images/ segment; done
```
