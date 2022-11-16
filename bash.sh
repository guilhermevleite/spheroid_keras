docker run -it --name=bash --gpus all --rm -v /home/leite/workspace:/workspace -w /workspace/spheroid_keras/unext -u $(id -u):$(id -g) phd_torch:1.1 bash
