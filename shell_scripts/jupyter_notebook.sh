docker run -it -p 8888:8888 --runtime=nvidia --rm -v $PWD:/home/leite/Workspace/ -w /home/leite/ -u $(id -u):$(id -g) phd jupyter notebook --ip 0.0.0.0 --no-browser
