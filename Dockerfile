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

ENV MPLCONFIGDIR=/home/leite/.config/matplotlib
