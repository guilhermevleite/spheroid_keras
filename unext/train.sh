idx=0
python3 train.py --dataset tiny --arch TransUnet --name trans_tiny --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --b 2

idx=0
python3 train.py --dataset tiny --arch Unet --name unet_tiny --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --b 2

idx=0
python3 train.py --dataset tiny --arch Unext --name unext_tiny --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --b 2

idx=0
python3 train.py --dataset tiny --arch MultiResUnet --name multi_tiny --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --b 2

idx=0
python3 train.py --dataset tiny --arch UnetPP --name unetpp_tiny --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --b 2
