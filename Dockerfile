FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
MAINTAINER Guilherme (guilherme.vieira.leite@gmail.com)

RUN pip3 install opencv-python \
                scikit-image \
                scikit-learn \
                pandas \
                albumentations \
                matplotlib \
                timm \
                mmcv \
                monai==0.7.0 \
                einops \
                ml_collections

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

ENV MPLCONFIGDIR=/home/leite/.config/matplotlib
