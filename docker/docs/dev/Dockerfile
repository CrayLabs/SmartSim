FROM ubuntu:20.04

LABEL maintainer="Cray Labs"

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Seattle

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    git gcc make \
    python3-pip python3 python3-dev cmake pandoc doxygen \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . /usr/local/src/SmartSim/
WORKDIR /usr/local/src/SmartSim/

# Install docs dependencies and SmartSim
RUN python -m pip install -r doc/requirements-doc.txt && \
    NO_CHECKS=1 SMARTSIM_SUFFIX=dev python -m pip install .

# Install smartredis
RUN git clone https://github.com/CrayLabs/SmartRedis.git --branch develop --depth=1 smartredis \
    && cd smartredis \
    && python -m pip install . \
    && rm -rf ~/.cache/pip

RUN cd doc/tutorials/ && \
    ln -s ../../tutorials/* .

RUN make docs
