#bin/bash

# when --cont is set, --epochs has to be higher than the number of epochs executed when the model has been saved the last time (this information is written and retrieved from the weights file name)
# ----------------------------------------------------------------------------------------------------
mkdir -p experiments/$1;
CUDA_VISIBLE_DEVICES=$2 python new_main.py --dataset flash --exp_name flash --base_path flash_GPS_Image_LiDAR --save_path experiments/$1/ --load-model '' --load-model-pruned '' --classes 64  --sparsity-type irregular --epochs 300 --epochs-prune 300 --epochs-mask-retrain 300 --admm-epochs 3 --mask-admm-epochs 10  --rho 0.01  --mixup --alpha 0 --smooth --smooth-eps 0.1 --config-setting 5,3,2 --adaptive-mask False --adaptive-ratio 0 --arch flashnet --depth 10 --tasks 3  \
> /home/anda/ThuanMin/Omni-CNN/experiments/$1/$1.out \
2> /home/anda/ThuanMin/Omni-CNN/experiments/$1/log.err
