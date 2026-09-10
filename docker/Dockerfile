# Build from the repository root:
# docker build --platform linux/amd64 -f docker/Dockerfile -t bevfusion-train:cu113 .
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=4
RUN test "$(uname -m)" = "x86_64" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.8 python3.8-dev python3.8-venv \
       build-essential ninja-build git ca-certificates \
       libgl1 libglib2.0-0 libgomp1 openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /opt/bevfusion
ENV PATH="/opt/bevfusion/bin:/usr/local/cuda/bin:${PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME="/workspace/cache/torch" \
    MPLCONFIGDIR="/workspace/cache/matplotlib" \
    NUMBA_CACHE_DIR="/workspace/cache/numba"

WORKDIR /workspace/bevfusion
COPY docker/requirements-cuda.txt docker/requirements-training.txt /opt/bevfusion-requirements/
RUN python -m pip install --no-cache-dir \
       pip==24.0 setuptools==59.5.0 wheel==0.38.4 \
    && python -m pip install --no-cache-dir --no-deps \
       -r /opt/bevfusion-requirements/requirements-cuda.txt \
    && python -m pip install --no-cache-dir \
       --only-binary=numba,llvmlite --no-binary=mpi4py \
       -r /opt/bevfusion-requirements/requirements-training.txt \
    && python -m pip check

# .dockerignore excludes host-built .so, data, weights, runs and Git metadata.
COPY . .
# Builds usually have no GPU. FORCE_CUDA prevents CPU-only extension builds.
# setup.py explicitly targets sm_70/75/80/86; this is not a Thor image.
RUN FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS} \
       python -m pip install --no-cache-dir -e . --no-deps --no-build-isolation \
    && python docker/check_environment.py --strict \
    && python -m pip check \
    && mkdir -p /workspace/cache /workspace/bevfusion/runs

CMD ["bash"]
