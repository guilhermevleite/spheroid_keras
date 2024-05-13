for counter in $(seq 1 4); do
    python3 train.py --dataset ours --arch Unet --name _no_aug_${counter} --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 24
done

for counter in $(seq 1 4); do
    python3 train.py --dataset ours --arch Unext --name _no_aug_${counter} --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 198
done

for counter in $(seq 1 4); do
    python3 train.py --dataset ours --arch MultiResUnet --name _no_aug_${counter} --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:0 --b 14
done
