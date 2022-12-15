idx=2

# for idx in {0..9}; do
#    python3 train.py --dataset busi --arch UNext --name busi_${idx} --img_ext .png --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 6 --early_stopping 10
#done;

#python3 train.py --dataset isic_2018 --arch MultiResUnet --name ${idx}_multires_isic_2018_100ep --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

python3 train.py --dataset ours --arch MultiResUnet --name ${idx}_multires_ours_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset ours_filtered --arch MultiResUnet --name ${idx}_multires_ours_filtered_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset busi --arch MultiResUnet --name ${idx}_multires_busi_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6


#python3 train.py --dataset isic_2018 --arch UNext --name ${idx}_unext_isic_2018_100ep --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

python3 train.py --dataset ours --arch UNext --name ${idx}_unext_ours_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset ours_filtered --arch UNext --name ${idx}_unext_ours_filtered_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset busi --arch UNext --name ${idx}_unext_busi_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6


#python3 train.py --dataset isic_2018 --arch Unet --name ${idx}_unet_isic_2018_100ep --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

python3 train.py --dataset ours --arch Unet --name ${idx}_unet_ours_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset ours_filtered --arch Unet --name ${idx}_unet_ours_filtered_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset busi --arch Unet --name ${idx}_unet_busi_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6


#python3 train.py --dataset isic_2018 --arch UnetPP --name ${idx}_unetpp_isic_2018_100ep --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

python3 train.py --dataset ours --arch UnetPP --name ${idx}_unetpp_ours_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset ours_filtered --arch UnetPP --name ${idx}_unetpp_ours_filtered_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6

#python3 train.py --dataset busi --arch UnetPP --name ${idx}_unetpp_busi_100ep --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --b 6
