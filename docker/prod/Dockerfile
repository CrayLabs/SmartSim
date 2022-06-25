FROM ubuntu:20.04

LABEL maintainer="Cray Labs"
LABEL org.opencontainers.image.source https://github.com/CrayLabs/SmartSim

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Seattle

RUN useradd --system --create-home --shell /bin/bash -g root -G sudo craylabs && \
    apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    git gcc make git-lfs wget libopenmpi-dev openmpi-bin \
    python3-pip python3 python3-dev cmake \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home/craylabs
COPY --chown=craylabs:root ./tutorials/ /home/craylabs/tutorials/

USER craylabs
RUN python -m pip install smartsim[ml]==0.4.1 jupyter jupyterlab matplotlib && \
    echo "export PATH=/home/craylabs/.local/bin:$PATH" >> /home/craylabs/.bashrc && \
    export PATH=/home/craylabs/.local/bin:$PATH && \
    smart build --device cpu -v && \
    chown craylabs:root -R /home/craylabs/.local && \
    rm -rf ~/.cache/pip

# remove non-jupyter notebook tutorials
RUN rm -rf /home/craylabs/tutorials/ray
CMD ["/bin/bash", "-c", "PATH=/home/craylabs/.local/bin:$PATH /home/craylabs/.local/bin/jupyter lab --port 8888 --no-browser --ip=0.0.0.0"]
