FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT=/tmp/constraints-cu124.txt \
    MUJOCO_GL=egl

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

COPY setup.py ./
COPY docker/constraints-cu124.txt /tmp/constraints-cu124.txt
RUN python -m pip install --upgrade pip setuptools wheel

COPY . .
RUN python -m pip install \
      "prettytable" \
      "tqdm" \
      "tyro>=1.0.1" \
      "torchrunx>=0.3.4" \
      "warp-lang==1.12.0" \
      "mujoco==3.8.0" \
      "mujoco-warp==3.8.0" \
      "trimesh>=4.8.3" \
      "viser>=1.0.24" \
      "mediapy>=1.2.6" \
      "imageio-ffmpeg" \
      "clearml" \
      "numpy" \
      "tensordict" \
      "rsl-rl-lib==5.0.1" \
      "tensorboard>=2.20.0" \
      "onnxscript>=0.5.4" \
      "wandb>=0.22.3" \
      "scipy" \
    && python -m pip install --no-deps "mjlab==1.2.0" \
    && python -m pip install --no-deps -e .

CMD ["/bin/bash"]
