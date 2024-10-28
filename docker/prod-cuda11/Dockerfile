# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FROM ubuntu:22.04

LABEL maintainer="Cray Labs"
LABEL org.opencontainers.image.source https://github.com/CrayLabs/SmartSim

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Seattle

# Make basic dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    git gcc make git-lfs wget libopenmpi-dev openmpi-bin unzip \
    python3-pip python3 python3-dev cmake wget apt-utils

# # Install Cudatoolkit 11.8
ENV TERM="xterm"
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    chmod +x ./cuda_11.8.0_520.61.05_linux.run && \
    ./cuda_11.8.0_520.61.05_linux.run --silent --toolkit && \
    rm ./cuda_11.8.0_520.61.05_linux.run

# Install cuDNN 8.9.7
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn8_8.9.7.29-1+cuda11.8_amd64.deb && \
    dpkg -i libcudnn8_8.9.7.29-1+cuda11.8_amd64.deb && \
    rm ./libcudnn8_8.9.7.29-1+cuda11.8_amd64.deb

 # Install SmartSim and SmartRedis
 RUN pip install git+https://github.com/CrayLabs/SmartRedis.git && \
     pip install "smartsim @ git+https://github.com/CrayLabs/SmartSim.git"

 ENV CUDA_HOME="/usr/local/cuda/"
 ENV PATH="${PATH}:${CUDA_HOME}/bin"

 # Build ML Backends
 RUN smart build --device=gpu --onnx
