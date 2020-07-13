FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="atgm1113@gmail.com"
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt