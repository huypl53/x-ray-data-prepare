#!/bin/sh

declare -a ds=(
"2_Viem_thuc_quan"
"3_Viem_da_day_HP_am"
# "4_Viem_da_day_HP_duong"
"5_Ung_thu_thuc_quan"
"6_Ung_thu_da_day"
"7_Loet_HTT"
)

root_dir=$1
null_image_dir=$2

declare -a modes=(
#'yolov8' 
'yolov8_seg'
)

for mode in "${modes[@]}";
do
  for d in "${ds[@]}";
  do
    image_dir="$root_dir/$d"
    im_train_dir="$image_dir/$mode/images/train"
    lb_train_dir=${im_train_dir/images/labels}
    if [[ ! -d $im_train_dir ]];
    then
      break
    fi

    im_train_num=`ls $im_train_dir | wc -l`
    num_null_im=$(( im_train_num ))
    for im in $( find $null_image_dir -type f | shuf | head -n $num_null_im );
    do
      cp "$im" "$im_train_dir"
      f0="${im%.*}.txt" && f1=${f0/images/labels} && cp "$f1" "$lb_train_dir"
    done
  done
done
