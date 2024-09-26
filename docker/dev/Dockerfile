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

RUN useradd --system --create-home --shell /bin/bash -g root -G sudo craylabs && \
    apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    git gcc make git-lfs wget libopenmpi-dev openmpi-bin unzip \
    python3-pip python3 python3-dev cmake \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home/craylabs
RUN git clone https://github.com/CrayLabs/SmartRedis.git --branch develop --depth=1 smartredis \
    && chown -R craylabs:root smartredis \
    && cd smartredis \
    && python -m pip install .

COPY . /home/craylabs/SmartSim
RUN chown craylabs:root -R SmartSim
USER craylabs

RUN cd SmartSim && SMARTSIM_SUFFIX=dev python -m pip install .

RUN export PATH=/home/craylabs/.local/bin:$PATH && \
    echo "export PATH=/home/craylabs/.local/bin:$PATH" >> /home/craylabs/.bashrc && \
    python -m pip install jupyter jupyterlab "ipython<8" matplotlib && \
    smart clobber && \
    smart build --device cpu -v && \
    chown craylabs:root -R /home/craylabs/.local && \
    rm -rf ~/.cache/pip

CMD ["/bin/bash", "-c", "PATH=/home/craylabs/.local/bin:$PATH /home/craylabs/.local/bin/jupyter lab --port 8888 --no-browser --ip=0.0.0.0"]
