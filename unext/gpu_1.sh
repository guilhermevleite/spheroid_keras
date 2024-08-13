#!/bin/bash

# ARCH_LST=("TransUnet" "Unext" "UnetPP")
# BATCH_LST=("26" "98" "18")
#
# DEVICE="cuda:1"
#
# DATASET="ours_ALL_split"
# NAME="split"
#
# for i in ${!ARCH_LST[@]}; do
# 	for c in $(seq 1 4); do
# 		python3 train.py --dataset ${DATASET} --arch ${ARCH_LST[$i]} --name ${NAME} --replica $c --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device ${DEVICE} --b ${BATCH_LST[$i]} --early_stopping 5 -est 25
#
# 	done
# done

ARCH_LST=("SwinUnet")
BATCH_LST=("64")

DEVICE="cuda:1"

DATASET="ours_ALL_split"
NAME="split"

for i in ${!ARCH_LST[@]}; do
	for c in $(seq 1 4); do
		python3 train.py --dataset ${DATASET} --arch ${ARCH_LST[$i]} --name ${NAME} --replica $c --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 224 --input_h 224 --device ${DEVICE} --b ${BATCH_LST[$i]} --early_stopping 5 -est 25

	done
done
