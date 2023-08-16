FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:tpu
WORKDIR /
RUN apt-get update
RUN apt-get -y upgrade
RUN pip install pytest tf-nightly
RUN mkdir -p /xla

# Builds and configure sccache
ENV OPENSSL_INCLUDE_DIR /usr/include/openssl
ENV OPENSSL_LIB_DIR /usr/lib/x86_64-linux-gnu

ENV CARGO_HOME /opt/cargo
ENV RUSTUP_HOME /opt/rustup

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN . $CARGO_HOME/env && \
    git clone --recursive https://github.com/pytorch/sccache.git && \
    cd sccache && \
    cargo install --path . && \
    cd .. && \
    rm -rf sccache

ENV PATH $CARGO_HOME/bin:$PATH

ADD ./ /xla

WORKDIR /xla
RUN touch env
RUN touch xla_env
RUN touch default_credentials.json
ARG USE_MKLDNN=0
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
RUN bash .circleci/build.sh

WORKDIR /tmp/pytorch/xla

WORKDIR /
RUN git clone --quiet https://github.com/pytorch/vision.git "/vision"
WORKDIR /vision
RUN python setup.py install

WORKDIR /tmp/pytorch/xla

ENV PJRT_DEVICE=CPU
ENV XLA_STABLEHLO_COMPILE=1
ENTRYPOINT pytest test/stablehlo