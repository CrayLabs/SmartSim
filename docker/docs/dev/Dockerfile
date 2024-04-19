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

# Install smartredis
RUN git clone https://github.com/CrayLabs/SmartRedis.git --branch develop --depth=1 smartredis \
    && cd smartredis \
    && python -m pip install . \
    && rm -rf ~/.cache/pip

# Install smartdashboard
RUN git clone https://github.com/CrayLabs/SmartDashboard.git --branch develop --depth=1 smartdashboard \
    && cd smartdashboard \
    && python -m pip install . \
    && rm -rf ~/.cache/pip

# Install docs dependencies and SmartSim
RUN python -m pip install -r doc/requirements-doc.txt \
    && NO_CHECKS=1 SMARTSIM_SUFFIX=dev python -m pip install .

RUN make docs
