#!/usr/src/bash


main() {

  declare -a ds=( "2_Viem_thuc_quan" "3_Viem_da_day_HP_am" "5_Ung_thu_thuc_quan" "6_Ung_thu_da_day" "7_Loet_HTT" );

  declare -a tasks=( "detect" "segment" )

  for task in "${tasks[@]}";
  do
    for d in "${ds[@]}"; do 

    if [[ "${task}" == 'segment' ]]
    then
      yolo settings datasets_dir=../data/ weights_dir=./$d/weights runs_dir=./$d/runs && yolo "$task" train data=./datasets/$d-seg-coco.yaml model=yolov8m-seg.pt imgsz=640 epochs=500
    else
      yolo settings datasets_dir=../data/ weights_dir=./$d/weights runs_dir=./$d/runs && yolo "$task" train data=./datasets/$d-bbox-coco.yaml model=yolov8m.pt imgsz=640 epochs=500
    fi

    done
  done

}

main "$@"
