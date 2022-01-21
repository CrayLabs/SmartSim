FROM ubuntu:20.04

LABEL maintainer="Sam Partee"

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Seattle
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    git gcc make \
    python3-pip python3 python3-dev cmake git-lfs doxygen

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir /data
WORKDIR /data
RUN git clone https://github.com/CrayLabs/SmartSim.git --branch develop --depth=1 \
    && cd SmartSim \
    && NO_CHECKS=1 python -m pip install .[doc]


WORKDIR /data/SmartSim/
RUN git clone https://github.com/CrayLabs/SmartRedis.git --branch develop --depth=1 smartredis \
    && cd smartredis \
    && python -m pip install .

RUN make docs
