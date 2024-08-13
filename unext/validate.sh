for folder in ./*; do
	echo $folder

	python /workspace/spheroid_segmentation/unext/val.py --name $folder --dataset "ours_ALL_test"
done
