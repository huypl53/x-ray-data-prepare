#!/usr/bin/bash

declare -a ds=(
"./2_Viem_thuc_quan"
"./3_Viem_da_day_HP_am"
"./4_Viem_da_day_HP_duong"
"./5_Ung_thu_thuc_quan"
"./6_Ung_thu_da_day"
"./7_Loet_HTT"
)

imgsz=640

declare -a models=(
"best.pt"
"best.onnx"
)

for d in "${ds[@]}";
do
  yolo export model=./$d/runs/detect/train/weights/best.pt opset=12 format=onnx imgsz=640,640 int8

  for model in "${models[@]}";
  do
    yolo settings datasets_dir=../data/ weights_dir=./$d/weights runs_dir=./$d/runs && yolo detect val data=./datasets/$d-coco.yaml model=./$d/runs/detect/train/weights/$model imgsz=$imgsz
  done
done
