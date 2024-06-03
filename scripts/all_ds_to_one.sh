#! /bin/bash


declare -a ds=( 
  "2_Viem_thuc_quan"  
  "3_Viem_da_day_HP_am" 
  "5_Ung_thu_thuc_quan" 
  "6_Ung_thu_da_day" 
  "7_Loet_HTT" )

main() {
  root_dir=$1 # folder contains all datasets
  task=$2 # segment or detect
  target_dir=$3 # target_dir

  if [[ ! -d "${root_dir}"  ]]
  then
    echo "${root_dir} not existed!"
    exit 1
  fi


  sub_path='yolov8'
  if [[ $task == 'segment' ]]
  then 
    sub_path='yolov8_seg'
  elif [[ $task == 'detect' ]]
  then
    sub_path='yolov8'
  else
    echo 'invalid input task'
    exit 1
  fi


  declare -a modes=( 'train' 'val' )
  for mode in "${modes[@]}"
  do
    # create target directories
    tg_image_dir="$target_dir/${sub_path}/images/${mode}"
    tg_label_dir="$target_dir/${sub_path}/labels/${mode}"
    mkdir -p $tg_image_dir
    mkdir -p $tg_label_dir

    ds_counter=0
    for d in "${ds[@]}"
    do
      # copy images from src to target directory
      image_dir="$root_dir/$d/$sub_path/images/${mode}"
      for im in $(find $image_dir -type f)
      do
        f="${im##*/}"; p="${im%/*}"
        tg_image_path="${tg_image_dir}/${d}_${f}"
        cp $im $tg_image_path
      done

      # change label name according to order of ds
      label_dir="$root_dir/$d/$sub_path/labels/${mode}"
      for lb in $(find $label_dir -type f)
      do
        f="${lb##*/}"; p="${lb%/*}"
        tg_label_path="$tg_label_dir/${d}_${f}"
        sed -e "s/^0/$ds_counter/g" $lb > $tg_label_path
      done


      ds_counter=$(( $ds_counter + 1 ))
    done

  done
}

main "$@"
