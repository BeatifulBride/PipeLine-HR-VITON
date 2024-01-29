FROM nvidia/cuda:11.1.1-base

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC  \
    && apt update  \
    && apt list --upgradable  \
    && apt install -y sudo vim curl git make python3.9 python3-setuptools python3.9-dev libgl1-mesa-glx libglib2.0-0 build-essential  \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1  \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py  \
    && python3.9 get-pip.py  \
    && echo 'alias python="python3"' >> ~/.bashrc  \
    && echo 'alias pip="pip3.9"' >> ~/.bashrc

# Activate bashrc
RUN . ~/.bashrc

WORKDIR /app
COPY . /app

# package
ENV PATH=/root/.local/bin:$PATH
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install pycocotools
RUN pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
# EXPOSE 8000
# CMD ["uvicorn", "main:app", "--reload","--host", "0.0.0.0", "--port", "8000"]
