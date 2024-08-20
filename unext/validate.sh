for folder in ./*; do
	echo $folder

	python /workspace/spheroid_codes/segmentation/unext/val.py --name $folder --dataset "ours_test"
done
