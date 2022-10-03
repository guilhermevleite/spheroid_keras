idx=0
python3 train.py --dataset isic_2018 --arch UNet --name unet_isic_${idx} --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 6 --early_stopping 10

#for idx in {0..9}; do
    #python3 train.py --dataset isic_2018 --arch UNext --name isic_${idx} --img_ext .jpg --mask_ext .png --lr 0.0001 --epochs 400 --input_w 256 --input_h 256 --b 6 --early_stopping 10
#done;

#for idx in {0..9}; do
   #python3 val.py --name busi_${idx}
   #python3 val.py --name isic_${idx}
#done;
