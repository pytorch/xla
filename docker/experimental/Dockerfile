ARG python_version=3.8
ARG debian_version=buster

ARG cuda=0
ARG cuda_version=11.2
ARG cudnn_version=8.1.1.33
# Debian repo doesn't have libcudnn
ARG cuda_repo=ubuntu1804
ARG tf_cuda_compute_capabilities="7.0,7.5,8.0"

ARG tpuvm=1
ARG build_cpp_tests=0
ARG package_version=1.14.0

ARG bazel_jobs=

# Contains all build dependencies for PT/XLA
FROM python:${python_version}-${debian_version} AS builder

RUN apt-get update
RUN apt-get install -y curl gnupg libomp5 libopenblas-dev
RUN pip install numpy pyyaml

ARG cuda
ARG cuda_version
ARG cudnn_version
ARG cuda_repo
COPY docker/experimental/maybe_install_cuda.sh .
RUN CUDA=${cuda} CUDA_REPO=${cuda_repo} CUDA_VERSION=${cuda_version} CUDNN_VERSION=${cudnn_version} ./maybe_install_cuda.sh build && rm maybe_install_cuda.sh
ENV LD_LIBRARY_PATH=${cuda:+"${LD_LIBRARY_PATH}:/usr/local/cuda-${cuda_version}/targets/x86_64-linux/lib:/usr/local/nvidia/lib64"}

ARG debian_version
RUN echo "deb http://apt.llvm.org/${debian_version}/ llvm-toolchain-${debian_version}-8 main" >> /etc/apt/sources.list
RUN echo "deb-src http://apt.llvm.org/${debian_version}/ llvm-toolchain-${debian_version}-8 main" >> /etc/apt/sources.list
RUN curl https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

RUN apt-get update
RUN apt-get -y install clang-8 clang++-8
RUN update-alternatives --install /usr/bin/clang++ clang++ $(which clang++-8) 70

ENV CC=clang-8
ENV CXX=clang++-8

RUN pip install mkl mkl-include setuptools typing_extensions cmake requests

RUN git clone --recursive --depth=1 https://github.com/pytorch/pytorch.git
WORKDIR /pytorch
COPY torch_patches/ torch_patches/
RUN bash -c "torch_pin=$(cat torch_patches/.torch_pin &); git fetch origin ${torch_pin:-master}; git checkout origin/${torch_pin:-master}"

# Disable CUDA for PyTorch
ENV USE_CUDA "0"

ARG package_version
RUN PYTORCH_BUILD_VERSION=${package_version} PYTORCH_BUILD_NUMBER=1 python setup.py bdist_wheel

RUN pip install dist/*.whl

RUN curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
RUN mv bazelisk-linux-amd64 /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel

RUN mkdir xla
WORKDIR /pytorch/xla/

# Contains actual build artifacts
FROM builder AS artifacts

COPY tf_patches/ tf_patches/
COPY third_party/ third_party/

RUN for p in tf_patches/*.diff; do patch -d third_party/tensorflow -N -p1 < $p; done

COPY build_torch_xla_libs.sh .

ARG tpuvm
ARG tf_cuda_compute_capabilities
ARG bazel_jobs
RUN TPUVM_MODE=${tpuvm} XLA_CUDA=${cuda} BAZEL_JOBS=${bazel_jobs} TF_CUDA_COMPUTE_CAPABILITIES=${tf_cuda_compute_capabilities} bash build_torch_xla_libs.sh -O -D_GLIBCXX_USE_CXX11_ABI=1

COPY torch_xla/ torch_xla/
COPY setup.py .
COPY xla_native_functions.yaml .

COPY scripts/ scripts/

ARG build_cpp_tests
ARG package_version
RUN TORCH_XLA_VERSION=${package_version} BUILD_CPP_TESTS=${build_cpp_tests} TPUVM_MODE=${tpuvm} BUNDLE_LIBTPU=${tpuvm} XLA_CUDA=${cuda} TF_CUDA_COMPUTE_CAPABILITIES=${tf_cuda_compute_capabilities} python setup.py bdist_wheel

RUN pip install dist/*.whl

# Contains only release artifacts
FROM python:${python_version}-slim-${debian_version} AS release

RUN apt-get update
RUN apt-get install -y libopenblas-base curl gnupg libomp5 git

RUN pip install numpy pyyaml

ARG cuda
ARG cudnn_version
ARG cuda_version
ARG cuda_repo

COPY docker/experimental/maybe_install_cuda.sh .
RUN CUDA=${cuda} CUDA_REPO=${cuda_repo} CUDA_VERSION=${cuda_version} CUDNN_VERSION=${cudnn_version} ./maybe_install_cuda.sh common && rm maybe_install_cuda.sh
ENV LD_LIBRARY_PATH=${cuda:+"${LD_LIBRARY_PATH}:/usr/local/cuda-${cuda_version}/targets/x86_64-linux/lib:/usr/local/nvidia/lib64"}

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >> /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update
RUN apt-get -y install google-cloud-cli

COPY --from=artifacts /pytorch/dist/*.whl .
COPY --from=artifacts /pytorch/xla/dist/*.whl .

RUN pip install *.whl
RUN rm *.whl

ARG tpuvm
RUN if [ "${tpuvm}" = "1" ]; then pip install torch_xla[tpuvm]; fi

# TODO: make this version selectable
RUN pip install --no-deps torchvision pillow

RUN mkdir -p /pytorch/xla
COPY . /pytorch/xla

WORKDIR /pytorch/xla

# TODO: this fails without a git repository initialized for some reason
RUN python setup.py clean || exit 0

WORKDIR /
