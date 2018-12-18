!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

# Pack tarball artifact with source
mkdir wheels_source
tar -czvf pytorch-tpu-nightly.tar.gz github/xla
cp pytorch-tpu-nightly.tar.gz wheels_source
cp pytorch-tpu-nightly.tar.gz wheels_source/"pytorch-tpu-$(date -d "yesterday" +%Y%m%d).tar.gz"

# Place pytorch/xla under pytorch/pytorch
cd github
git clone https://github.com/pytorch/pytorch.git
mv xla/ pytorch/
cd pytorch
git submodule update --init --recursive

# TODO(jysohn): remove following patching once pytorch JIT bug is fixed
git checkout d71fac20ebf6b393e464d312b8a38ae1c45c3386
git apply xla/pytorch.patch

# Execute build
cd xla
./kokoro_build.sh
