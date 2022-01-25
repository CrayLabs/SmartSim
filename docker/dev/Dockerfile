FROM ubuntu:20.04

LABEL maintainer="Cray Labs"
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Seattle

#TODO
# - add openmpi
# - add tutorials from other sections
# - add permissions to the craylabs user

RUN useradd --create-home --shell /bin/bash craylabs

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    git gcc make git-lfs wget \
    python3-pip python3 python3-dev cmake \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /home/craylabs
USER craylabs

RUN git clone https://github.com/CrayLabs/SmartRedis.git --branch develop --depth=1 smartredis \
    && cd smartredis \
    && python -m pip install .


COPY . /home/craylabs/SmartSim
RUN cd SmartSim && NO_CHECKS=1 SMARTSIM_SUFFIX=dev python -m pip install .

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
    smart build --device cpu -v --no_tf

SHELL ["bash", "-c"]
CMD ["/home/craylabs/.local/bin/jupyter", "lab", "--port", "8888", "--no-browser", "--ip=0.0.0.0"]