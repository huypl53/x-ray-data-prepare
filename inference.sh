#!/usr/bin/bash

# bash ./inference.sh ./path/to/model.pt ./path/to/image image_size

main(){
  model_path=$1
  image_path=$2
  image_size=$3
  if [[ -z $image_size ]]
  then
    image_size=640
  fi

  yolo predict model="$model_path" source="$image_path" "$image_size"

}

main "$@"
