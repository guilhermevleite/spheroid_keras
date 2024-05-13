for counter in $(seq 1 4); do
	python3 train.py --dataset ours --arch TransUnet --name _no-aug_${counter} --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:1 --b 26
done

for counter in $(seq 1 4); do
    python3 train.py --dataset ours --arch UnetPP --name _no-aug_${counter} --img_ext .png --mask_ext .png --lr 0.0001 --epochs 100 --input_w 256 --input_h 256 --device cuda:1 --b 18
done
