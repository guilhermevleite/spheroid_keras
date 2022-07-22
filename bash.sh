docker run -it --name=bash --runtime=nvidia --rm -v /home/leite/Workspace:/workspace -w /workspace -u $(id -u):$(id -g) phd bash
