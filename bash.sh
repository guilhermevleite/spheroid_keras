docker run -it --name=bash --gpus all --rm -v /home/leite/Workspace:/workspace -w /workspace/spheroid_keras -u $(id -u):$(id -g) phd_torch bash
