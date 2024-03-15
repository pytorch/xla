FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:tpu
WORKDIR /
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install clang time
RUN pip install pytest
ARG USE_MKLDNN=0
ARG SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2
ARG XLA_CUDA=0
ARG BAZEL_REMOTE_CACHE=1
ARG USE_FBGEMM=0
ARG USE_OPENMP=0
ARG BUILD_NVFUSER=0
ARG USE_KINETO=0
ARG USE_NCCL=0
ARG USE_XNNPACK=0
ARG USE_QNNPACK=0
ARG USE_PYTORCH_QNNPACK=0
ARG USE_CUDA=0
ARG BUILD_TEST=0
ARG USE_CACHE=1

# install pytorch
ARG PYTORCH_DIR=/pytorch
ARG XLA_DIR=$PYTORCH_DIR/xla
RUN git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"

WORKDIR /pytorch
RUN pip install cmake ninja
RUN pip install  -r requirements.txt
RUN bash .ci/docker/common/install_cache.sh
RUN time python setup.py install
RUN /opt/cache/bin/sccache --show-stats

# install torchvision
WORKDIR /
RUN git clone --quiet https://github.com/pytorch/vision.git "/vision"
WORKDIR /vision
RUN python setup.py install

# install torch xla
ADD ./ /pytorch/xla
WORKDIR /pytorch/xla
RUN cp keystore_content/77422_pytorch_tpu_cloud_build default_credentials.json
ARG SCCACHE="$(which sccache)"

WORKDIR /pytorch/xla
ARG GCLOUD_SERVICE_KEY_FILE="/pytorch/xla/default_credentials.json"
ARG SILO_NAME='cache-silo-ci-dev-3.8_cuda_12.1'  # cache bucket for CI
RUN time pip install -e .

# Run tests
ENV PJRT_DEVICE=CPU
ENV XLA_STABLEHLO_COMPILE=1
ENTRYPOINT pytest test/stablehlo
