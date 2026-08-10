FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl

ENV PATH="${CONDA_DIR}/bin:${PATH}"

WORKDIR /workspace/unitree_rl_mjlab

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
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

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && conda install -y python=3.11 \
    && conda clean -afy

COPY setup.py ./
RUN python -m pip install --upgrade pip setuptools wheel

COPY . .
RUN python -m pip install -e .

CMD ["/bin/bash"]
