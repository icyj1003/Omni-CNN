#bin/bash

# when --cont is set, --epochs has to be higher than the number of epochs executed when the model has been saved the last time (this information is written and retrieved from the weights file name)
# ----------------------------------------------------------------------------------------------------
mkdir -p experiments/$1;
CUDA_VISIBLE_DEVICES=$2 python train.py --dataset flash --exp_name flash --base_path flash_GPS_Image_LiDAR --save_path experiments/$1/ --load-cummu-model experiments/$1/flash/task2/cumu_model.pt --classes 64 --epochs 300  --mixup --alpha 0 --smooth --smooth-eps 0.1 --config-setting 3,5,2 --arch flashnet --depth 10 --tasks 3 --learning-mode centralized \
> /home/anda/ThuanMin/Omni-CNN/experiments/$1/log.out \
2> /home/anda/ThuanMin/Omni-CNN/experiments/$1/log.err
