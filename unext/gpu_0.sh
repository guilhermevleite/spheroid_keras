#!/bin/bash

# ARCH_LST=("UnetAtt" "Unet" "MultiResUnet")
# BATCH_LST=("24" "24" "14")

# ARCH_LST=("Unet")
# BATCH_LST=("24")
#
DEVICE="cuda:0"
#
# DATASET="ours_train_AUG"
# NAME="patched"
#
# for i in ${!ARCH_LST[@]}; do
# 	for c in $(seq 1 4); do
# 		python3 train.py --dataset ${DATASET} --arch ${ARCH_LST[$i]} --name ${NAME} --replica $c --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 224 --input_h 224 --device ${DEVICE} --b ${BATCH_LST[$i]} --early_stopping 5 -est 25
# 	done
# done

ARCH_LST=("SwinUnet")
BATCH_LST=("64")
BATCH_LST=("2")

DATASET="ours_train_AUG"
DATASET="tiny"
NAME="swin_param_1"

HEAD_COUNT=4
PATCH_SIZE=4
SWINDOW_SIZE=3

for i in ${!ARCH_LST[@]}; do
	for c in $(seq 1 4); do
		python3 train.py --dataset ${DATASET} --arch ${ARCH_LST[$i]} --name ${NAME} --replica $c --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 224 --input_h 224 --device ${DEVICE} --b ${BATCH_LST[$i]} --early_stopping 5 -est 25 --T_head_count ${HEAD_COUNT} --T_patch_size ${PATCH_SIZE} --S_swindow_size ${SWINDOW_SIZE}
		break
	done
done
