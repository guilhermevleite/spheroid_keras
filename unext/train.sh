idx=0

# for idx in {0..9}; do
#    python3 train.py --dataset busi --arch UNext --name busi_${idx} --img_ext .png --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 6 --early_stopping 10
#done;

#for idx in {0..9}; do
#    python3 train.py --dataset isic_2018 --arch UNext --name isic_${idx} --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 6 --early_stopping 10
#done;

#for idx in {0..9}; do
#   python3 val.py --name busi_${idx}
#   python3 val.py --name isic_${idx}
#done;

#python3 train.py --dataset isic_2018 --arch UNext --name ${idx}_isic --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 6 --early_stopping 10

#python3 fine.py --dataset busi --arch UNext --name ${idx}_isic --tune busi --img_ext .png --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 6 --early_stopping 10

#python3 train.py --dataset busi --arch UNext --name ${idx}_busi --img_ext .png --mask_ext .png --lr 0.0001 --epochs 28 --input_w 256 --input_h 256 --b 6

#python3 fine.py --dataset busi --arch UNext --name ${idx}_isic --tune busi --img_ext .png --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 6 --early_stopping 10

#python3 train.py --dataset isic_2018 --arch Unet --name ${idx}_unet_isic_200ep --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 200 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset cpu --arch UNETR --name ${idx}_unetr_cpu_2ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 2 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset isic_2018 --arch UNext --name ${idx}_unext_isic_200ep --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 200 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset busi --arch UNext --name ${idx}_unet_isic_200ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 200 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset isic_2018 --arch TransUNet --name ${idx}_tunet_isic_100ep --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset busi --arch TransUNet --name ${idx}_tunet_busi_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset ours_filtered --arch TransUNet --name ${idx}_tunet_ours_filtered_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

python3 train.py --dataset tiny --arch Unet --name unet_tiny_test_env --img_ext .png --mask_ext .png --lr 0.0001 --epochs 1 --input_w 256 --input_h 256 --b 2
