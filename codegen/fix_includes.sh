#!/bin/bash

# torchgen insists on including the tensor header as a system <tensor.h> header
sed -i 's#<torch_xla/csrc/tensor.h>#"torch_xla/csrc/tensor.h"#' $@

# remove the runfiles-prefix used in codegen for pytorch
# `torchgen` generates relative includes path and does not support customizing the root,
# so we have to fix them up.
sed -i 's#bazel-out/k8-opt/bin/codegen/lazy_tensor_generator.runfiles/torch/##' $@

# use the generated files that are in the compilation unit
sed -i 's#<bazel-out/\(.*\)>#"bazel-out/\1"#' $@
