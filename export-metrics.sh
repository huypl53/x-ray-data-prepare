#!/usr/bin/bash

declare -a ds=(
"./2_Viem_thuc_quan"
"./3_Viem_da_day_HP_am"
"./4_Viem_da_day_HP_duong"
"./5_Ung_thu_thuc_quan"
"./6_Ung_thu_da_day"
"./7_Loet_HTT"
)

task=$1

if [[ "$task" != "detect" && "$task" != "segment" ]];
then
  echo "Task must be detect or segment"
  exit

fi

_suffix=''
if [[ "$task" == "segment" ]];
then
  _suffix='-seg'
fi

if [[ "$task" == "detect" ]]
then 
  _suffix='-bbox'
fi

imgsz=640

declare -a models=(
# "best.pt"
# "best.onnx"
)

declare -a openvinoModels=(
"float32_best_openvino_model"
"float16_best_openvino_model"
# "int8_best_openvino_model"
)

for d in "${ds[@]}";
do
  # for model in "${models[@]}";
  # do
  #   yolo settings datasets_dir=../data/ weights_dir=./$d/weights runs_dir=./$d/runs && python metrics.py ./$d/runs/detect/train/weights/$model ./datasets/$d-coco.yaml
      # yolo detect val data=./datasets/$d-coco.yaml model=./$d/runs/detect/train/weights/$model imgsz=$imgsz
  # done

  for model in "${openvinoModels[@]}";
  do
    args=''
    if [[ $model =~ "float16" ]];
    then
      args='half'
    elif [[ $model =~ "int8" ]];
    then
      args='int8'
    fi
    yolo export model=./$d/runs/$task/train/weights/best.pt format=openvino imgsz=640,640 $args
    rm -rf  ./$d/runs/$task/train/weights/$model/
    mv ./$d/runs/$task/train/weights/best_openvino_model/  ./$d/runs/$task/train/weights/$model/

    yolo settings datasets_dir=../data/ weights_dir=./$d/weights runs_dir=./$d/runs && python metrics.py ./$d/runs/$task/train/weights/$model ./datasets/$d$_suffix-coco.yaml $task
  done
done
