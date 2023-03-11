# Use nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 or later for CUDA.
# Warning: All ARGs placed before FROM will only be scoped up unitl FROM statement.
# https://github.com/docker/cli/blob/3c7ede6a68941f64c3a154c9a753eb7a9b1c2c3e/docs/reference/builder.md#understand-how-arg-and-from-interact
ARG base_image="debian:buster"
FROM "${base_image}"

ARG python_version="3.7"
ARG release_version="nightly"
ARG xla_branch=""
ARG example_branch="master"
ARG cuda="0"
ARG cuda_compute="3.7,7.0,7.5,8.0"
ARG cxx_abi="1"
ARG tpuvm=""
ARG bazel_jobs=""
ARG git_clone="true"

ENV BUILD_CPP_TESTS "${build_cpp_tests}"

RUN apt-get update
RUN apt-get install -y git sudo python3-pip
# To enable `base_image=ubuntu:focal`
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN git clone https://github.com/pytorch/pytorch

# Disable CUDA for PyTorch and ensure the pre-built wheel works
# on Colab with no libmpi installed.
ENV USE_CUDA "0"
ENV USE_MPI "0"

# Enable CUDA for XLA
ENV XLA_CUDA "${cuda}"
ENV TF_CUDA_COMPUTE_CAPABILITIES "${cuda_compute}"

# Whether to build for TPUVM mode
ENV TPUVM_MODE "${tpuvm}"
ENV BUNDLE_LIBTPU "${tpuvm}"

# Maximum number of jobs to use for bazel build
ENV BAZEL_JOBS "${bazel_jobs}"

# To get around issue of Cloud Build with recursive submodule update
# clone recursively from pytorch/xla if building docker image with
# cloud build. Otherwise, just use local.
# https://github.com/GoogleCloudPlatform/cloud-builders/issues/435
COPY . /pytorch/xla
RUN if [ "${git_clone}" = "true" ]; then \
  if [ -z "${xla_branch}" ]; then xla_branch="${release_version}" && example_branch="${release_version}"; fi && \
  if [ "${xla_branch}" = "nightly" ]; then xla_branch="master" && example_branch="master"; fi && \
  echo "\nxla_branch: ${xla_branch} example_branch: ${example_branch} \n" && \
  cd /pytorch && \
  rm -rf xla && \
  git clone -b "${xla_branch}" --recursive https://github.com/pytorch/xla && \
  cd / && \
  git clone -b "${example_branch}" --recursive https://github.com/pytorch-tpu/examples tpu-examples; fi

RUN cd /pytorch/xla && bash scripts/build_torch_wheels.sh ${python_version} ${release_version}

# Set LD_PRELOAD to use tcmalloc if for tpuvm mode.
ENV LD_PRELOAD=${tpuvm:+"/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"}

# Use conda environment on startup or when running scripts.
RUN echo "conda activate pytorch" >> ~/.bashrc
RUN echo "export TF_CPP_LOG_THREAD_ID=1" >> ~/.bashrc
ENV PATH /root/anaconda3/envs/pytorch/bin/:/root/bin:$PATH

# Define entrypoint and cmd
COPY docker/docker-entrypoint.sh /usr/local/bin
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
