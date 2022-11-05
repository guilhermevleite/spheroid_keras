docker run -it --name=bash --gpus all --rm -v /home/leite/Workspace:/workspace -v /home/leite/sync/desktop_lx/db:/datasets -w /workspace -u $(id -u):$(id -g) phd_torch bash
