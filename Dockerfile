# Latest version of Debian
FROM debian:latest

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install packages
RUN apt-get -y update && apt-get install --no-install-recommends -y     \
    bzip2 curl unzip perl wget tcsh rpm         \
    git cmake python3 python3-pip                   \
    vim emacs autoconf automake libtool             \
    python3-setuptools make g++ gcc                 \
    uuid-dev libcurl4-gnutls-dev

# Get wget
RUN apt-get install wget

RUN mkdir /opt/smartsim
WORKDIR /opt/smartsim

# get and setup SmartSim
ADD SmartSim-0.1.0.tar.gz .
RUN cd SmartSim-0.1.0 && pip3 install -r requirements.txt && /bin/bash setup_env.sh

# run the synthetic simulation
RUN cd SmartSim-0.1.0 && /bin/bash setup_env.sh
ENV PYTHONPATH=$PYTHONPATH:/opt/smartsim/SmartSim-0.1.0/
ENV PATH=$PATH:/opt/smartsim/SmartSim-0.1.0/third-party/keydb/src/
