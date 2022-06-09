# syntax=docker/dockerfile:1
FROM ubuntu:21.10
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install -y python3 python3-pip python-is-python3 cmake git
RUN pip install torch==1.9.1
RUN git clone https://github.com/CrayLabs/SmartRedis.git
RUN cd SmartRedis && pip install . && make lib; cd ..

