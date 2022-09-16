ARG python_version=3.8
ARG debian_version=buster
FROM python:${python_version}-${debian_version} AS common

RUN apt-get update
RUN apt-get install -y curl gnupg libomp5
RUN pip install numpy pyyaml

ARG debian_number=ubuntu1804

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${debian_number}/x86_64/3bf863cc.pub
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/${debian_number}/x86_64/ /" >> /etc/apt/sources.list
RUN apt-get update

ARG cudnn_version="8.1.1.33"
ARG cuda_version="11.2"

RUN apt-get install -y libcudnn8=8.1.1.33-1+cuda11.2

FROM common AS builder

ARG debian_version=buster
ARG cudnn_version="8.1.1.33"
ARG cuda_version="11.2"

RUN apt-get install -y libopenblas-dev
RUN apt-get install -y libcudnn8-dev=${cudnn_version}-1+cuda${cuda_version}

RUN echo "deb http://apt.llvm.org/${debian_version}/ llvm-toolchain-${debian_version}-8 main" >> /etc/apt/sources.list
RUN echo "deb-src http://apt.llvm.org/${debian_version}/ llvm-toolchain-${debian_version}-8 main" >> /etc/apt/sources.list
RUN curl https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -

RUN apt-get update
RUN apt-get -y install clang-8 clang++-8
RUN update-alternatives --install /usr/bin/clang++ clang++ $(which clang++-8) 70

ENV CC=clang-8
ENV CXX=clang++-8

RUN git clone --recursive --depth=1 https://github.com/pytorch/pytorch.git
WORKDIR /pytorch

RUN pip install mkl mkl-include setuptools typing_extensions cmake

RUN USE_CUDA=0 python setup.py bdist_wheel

RUN pip install dist/*.whl

RUN curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
RUN mv bazelisk-linux-amd64 /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel

RUN mkdir xla
COPY third_party xla/third_party
COPY build_torch_xla_libs.sh xla/
WORKDIR /pytorch/xla/

RUN apt-get install -y cuda-toolkit-11-2
RUN XLA_CUDA=1 bash build_torch_xla_libs.sh -O -D_GLIBCXX_USE_CXX11_ABI=1

COPY torch_xla torch_xla
COPY scripts scripts
COPY setup.py .
COPY xla_native_functions.yaml .

RUN BUILD_CPP_TESTS=0 XLA_CUDA=1 python setup.py bdist_wheel

RUN pip install dist/*.whl

FROM common AS release

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >> /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update
RUN apt-get -y install google-cloud-cli

COPY --from=builder /pytorch/dist/*.whl .
COPY --from=builder /pytorch/xla/dist/*.whl .

RUN pip install *.whl
RUN pip install torch_xla[tpuvm]
RUN rm *.whl
