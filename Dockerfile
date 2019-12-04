# Latest version of Debian
FROM debian:latest

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install packages 
RUN apt-get -y update && apt-get install --no-install-recommends -y     \
    bzip2 curl unzip perl wget tcsh rpm         \
    git cmake python python3-pip                   \
    vim emacs

# Get wget
RUN apt-get install wget

RUN mkdir ~/.ss
WORKDIR ~/.ss

# get and setup SmartSim
ADD dist/SmartSim-0.1.0.tar.gz .
RUN cd SmartSim-0.1.0 && pip3 install -r requirements.txt && /bin/bash setup_env.sh

# Start the bash
RUN chmod a+rwx /etc/bash.bashrc