FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
MAINTAINER Guilherme (guilherme.vieira.leite@gmail.com)

RUN pip3 install opencv-python
RUN pip3 install scikit-image
RUN pip3 install scikit-learn
RUN pip3 install pandas
RUN pip3 install albumentations
RUN pip3 install matplotlib
RUN pip3 install timm
RUN pip3 install mmcv
RUN pip3 install monai=0.7.0

RUN mkdir -p /.config/matplotlib
