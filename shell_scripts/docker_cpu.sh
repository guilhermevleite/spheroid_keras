docker run -it --name=cpu_leite --ipc=host --rm -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /home/leite/workspace:/workspace -w /workspace/learning/spheroid_segmentation/unext -u $(id -u):$(id -g) phd_torch:1.2 sh ./cpu.sh
