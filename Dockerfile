FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl

WORKDIR /workspace/unitree_rl_mjlab

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libboost-all-dev \
    libegl1 \
    libeigen3-dev \
    libfmt-dev \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libgles2 \
    libsm6 \
    libspdlog-dev \
    libxext6 \
    libxrender1 \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py ./
RUN python -m pip install --upgrade pip setuptools wheel

COPY . .
RUN python -m pip install -e .

CMD ["/bin/bash"]
