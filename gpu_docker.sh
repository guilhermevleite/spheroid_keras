docker run -it --name=gpu_leite_2 --gpus all --ipc=host --rm -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /home/leite/workspace:/workspace -v /media/leite/data:/media -w /workspace/spheroid_segmentation/unext/ -u $(id -u):$(id -g) phd_torch bash
