idx=0
device='cuda:0'

python3 train.py --dataset tiny --arch Unet --name unet_tiny_test_env --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --b 2 --device ${device}

python3 train.py --dataset ours_filtered --arch Unet --name ${idx}_unet_ours_filtered_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6 --device ${device}
