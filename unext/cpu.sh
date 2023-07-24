python3 train.py --dataset tiny --arch TransUnet --name tiny_test --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --device cpu --b 2

# --- LACALLE DATASET
# python3 train.py --dataset lacalle_spheroidj --arch TransUnet --name friday_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 26
# 
# python3 train.py --dataset lacalle_spheroidj --arch TransUnet --name friday_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 200 --input_w 256 --input_h 256 --device cuda:0 --b 26
# 
# python3 train.py --dataset lacalle_spheroidj --arch Unet --name friday_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 224 --input_h 224 --device cuda:0 --b 24
# 
# python3 train.py --dataset lacalle_spheroidj --arch Unext --name friday_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 198
# 
# python3 train.py --dataset lacalle_spheroidj --arch MultiResUnet --name friday_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 14
#  
# python3 train.py --dataset lacalle_spheroidj --arch UnetPP --name friday_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 18
# 
# 
# # --- OUR DATASET
# python3 train.py --dataset ours_filtered --arch TransUnet --name friday_our-filtered --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 26

# python3 train.py --dataset ours_filtered --arch TransUnet --name friday_our-filtered --img_ext .png --mask_ext .png --lr 0.0001 --epochs 200 --input_w 256 --input_h 256 --device cuda:0 --b 26
# 
# python3 train.py --dataset ours_filtered --arch Unet --name friday_our-filtered --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 224 --input_h 224 --device cuda:0 --b 24
# 
# python3 train.py --dataset ours_filtered --arch Unext --name friday_our-filtered --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 64
# 
# python3 train.py --dataset ours_filtered --arch MultiResUnet --name friday_our-filtered --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 14
#  
# python3 train.py --dataset ours_filtered --arch UnetPP --name friday_our-filtered --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 18

# --- TRANSUNET Variation
# python3 train.py --dataset ours_filtered --arch TransUnet_h8 --name friday_our-filtered --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 18

# python3 train.py --dataset lacalle_spheroidj --arch TransUnet_h8 --name friday_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 18

