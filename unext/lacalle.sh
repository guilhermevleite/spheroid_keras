# --- LACALLE DATASET
# python3 train.py --dataset lacalle_spheroidj --arch TransUnet --name early_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --early_stopping 10 --b 26
# 
# python3 train.py --dataset lacalle_spheroidj --arch TransUnet --name early_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 200 --input_w 256 --input_h 256 --device cuda:0 --early_stopping 10 --b 26
# 
# python3 train.py --dataset lacalle_spheroidj --arch Unet --name early_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 224 --input_h 224 --device cuda:0 --early_stopping 10 --b 24
# 
# python3 train.py --dataset lacalle_spheroidj --arch Unext --name early_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --early_stopping 10 --b 198

python3 train.py --dataset lacalle_spheroidj --arch MultiResUnet --name early_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --early_stopping 10 --b 14
 
python3 train.py --dataset lacalle_spheroidj --arch UnetPP --name early_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --early_stopping 10 --b 18

# --- TRANSUNET Variation
# python3 train.py --dataset ours_filtered --arch TransUnet_h8 --name early_our-filtered --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --early_stopping 10 --b 18
# 
# python3 train.py --dataset lacalle_spheroidj --arch TransUnet_h8 --name early_spheroidj --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --early_stopping 10 --b 18
