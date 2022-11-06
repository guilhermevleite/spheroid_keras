WK_PATH="/home/leite/workspace"
DS_PATH="/home/leite/sync/lenovo/db"

docker run -it --name=bash --gpus all --rm -v $WK_PATH:/workspace -v $DS_PATH:/datasets -w /workspace -u $(id -u):$(id -g) phd_torch bash
