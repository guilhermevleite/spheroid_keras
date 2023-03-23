idx=0
python3 train.py --dataset tiny --arch Unet --name unet_tiny_test_env --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --b 2
