FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt update && \
    apt install -y sudo python3.7 python3-pip vim python3-opencv

RUN echo 'alias python="python3.7"' >> ~/.bashrc
RUN echo 'alias pip="pip3"' >> ~/.bashrc
RUN apt clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Activate bashrc
RUN . ~/.bashrc

# Install cuDNN 7.6.5
COPY cuda/include/cudnn* /usr/local/cuda-10.0/include
COPY cuda/lib64/* /usr/local/cuda-10.0/lib64/
RUN chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*

# Install requirements
COPY . /usr/www
WORKDIR /usr/www
RUN pip3 install --upgrade pip

# Support C.UTF-8
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8