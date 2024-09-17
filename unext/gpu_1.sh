#!/bin/bash

DEVICE="cuda:1"

# ARCH_LST=("UnetAtt" "Unet" "MultiResUnet" "SwinUnet")
# BATCH_LST=("24" "24" "14" "64")

# ARCH_LST=("TransUnet" "Unext" "UnetPP")
# BATCH_LST=("26" "98" "18")

ARCH_LST=("SwinUnet")
BATCH_LST=("32")
NAME="C_64"

DATASET="ours_train_AUG"

# Params
SIZE=224
S_C=64

for i in ${!ARCH_LST[@]}; do
	for c in $(seq 1 4); do
		python3 train.py --dataset ${DATASET} --arch ${ARCH_LST[$i]} --name ${NAME} --replica $c --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w ${SIZE} --input_h ${SIZE} --device ${DEVICE} --b ${BATCH_LST[$i]} --early_stopping 5 -est 25 --S_C ${S_C}
	done
done
