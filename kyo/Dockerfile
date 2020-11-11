FROM ubuntu:16.04
LABEL maintainer "Kyo Kutsuzawa <kutsuzawa@tohoku.ac.jp>"

# Add a ppa repository for python 3.6
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup python3.6
RUN apt-get update \
    && apt-get install -y \
        python3.6-dev \
        python3.6-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Change python command to run python 3.6
RUN update-alternatives --remove python /usr/bin/python2 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

# Get pip
RUN apt-get update \
    && apt-get install -y curl \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup python packages
RUN python -m pip install --no-cache-dir \
        pip==20.1.1 \
        enum34==1.1.10 \
    && python -m pip install --no-cache-dir \
        numpy==1.16.2 \
        scipy==1.1.0 \
        matplotlib==3.0.2 \
        gym==0.17.2

# Copy mujuco-related files
COPY mujocofiles/.mujoco /root/.mujoco
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin

# Setup mujoco-py and related packages
RUN apt-get update \
    && apt-get install -y \
        libosmesa6-dev \
        freeglut3-dev \
        libglu1-mesa-dev \
        patchelf \
        build-essential \
        libssl-dev \
        libffi-dev \
    && python -m pip install --no-cache-dir \
        cffi==1.14.0 \
        Cython==0.29.21 \
        lockfile==0.12.2 \
        glfw==1.12.0 \
        imageio==2.9.0 \
    && python -m pip install --no-cache-dir \
        mujoco-py==2.0.2.1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN python -m pip install --no-cache-dir \
        pandas \
        torch

# Install Spinningup
RUN apt-get update \
    && apt-get install -y \
        libopenmpi-dev \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/openai/spinningup.git \
    && python -m pip install --no-cache -e /spinningup

# Install development tools
RUN apt-get update \
    && apt-get install -y \
        vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Unset an environmental variable for display
RUN unset LIBGL_ALWAYS_INDIRECT

# Install additional packages
RUN python -m pip install --no-cache-dir \
        scikit-learn==0.23.1 \
        cma==3.0.3

RUN python -m pip install --no-cache-dir \
        cmaes==0.6.1

WORKDIR /root
