FROM ubuntu:20.04

LABEL maintainer="Cray Labs"
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Seattle

RUN useradd --system --create-home --shell /bin/bash -g root -G sudo craylabs

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    git gcc make git-lfs wget libopenmpi-dev openmpi-bin \
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

RUN cd SmartSim && \
    SMARTSIM_SUFFIX=dev python -m pip install .

# TODO set SmartRedis version when thats put in
#RUN git clone https://github.com/CrayLabs/SmartSim.git --branch develop --depth=1 \
#    && cd SmartSim \
#    && python -m pip install .[ml,ray]

# Jupyter
RUN python -m pip install ipython ipykernel \
    jupyter jupyterlab \
    && rm -rf ~/.cache/pip

RUN echo "export PATH=/home/craylabs/.local/bin:$PATH" >> /home/craylabs/.bashrc && \
    export PATH=/home/craylabs/.local/bin:$PATH && \
    smart clobber && \
    smart build --device cpu -v --no_tf

SHELL ["bash", "-c"]
CMD ["/home/craylabs/.local/bin/jupyter", "lab", "--port", "8888", "--no-browser", "--ip=0.0.0.0"]