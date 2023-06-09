"""Rules that simplify deps and compiler configuration for PyTorch/XLA."""

load(
    "//tensorflow/tsl/platform/default:rules_cc.bzl",
    "cc_test",
)

load(
    "//tensorflow/tsl:tsl.bzl",
    "clean_dep",
)

load(
    "//tensorflow/core/platform:build_config_root.default.bzl",
    "if_dynamic_kernels"
)

def if_dynamic_kernels(extra_deps, otherwise = []):
    return select({
        str(Label("//tensorflow:dynamic_loaded_kernels")): extra_deps,
        "//conditions:default": otherwise,
    })

def ptxla_cc_library(
        deps = [],
        copts = [],
        **kwargs):
    native.cc_library(
        copts = copts + ["-isystemexternal/torch"],  # Required for system includes.
        deps = deps + [
            "@torch//:headers",
            "@torch//:runtime_headers",
        ],
        **kwargs
    )

# Helper functions to add kernel dependencies to tf binaries when using static
# kernel linking.
def tf_binary_dynamic_kernel_deps(kernels):
    return if_dynamic_kernels(
        extra_deps = [],
        otherwise = kernels,
    )

def if_mkl_ml(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with MKL-ML.

    Args:
      if_true: expression to evaluate if building with MKL-ML.
      if_false: expression to evaluate if building without MKL-ML
        (i.e. without MKL at all, or with MKL-DNN only).

    Returns:
      a select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@org_tensorflow//third_party/mkl_dnn:build_with_mkl_opensource": if_false,
        "@org_tensorflow//tensorflow/tsl/mkl:build_with_mkl": if_true,
        "//conditions:default": if_false,
    })

def ptxla_cc_test(
        deps,
        copts = [],
        **kwargs):
    cc_test(
        copts = copts + [
            "-isystemexternal/torch",  # Required for system includes.
            "-fexceptions",  # Required for testing crashes.
        ],
        deps = deps + [
            "@pybind11//:pybind11_embed",  # libpython
            "@torch//:headers",
            "@torch//:libc10",
            "@torch//:libtorch",
            "@torch//:libtorch_cpu",
            "@torch//:libtorch_python",
        ] + tf_binary_dynamic_kernel_deps([]) + if_mkl_ml(
            [
                clean_dep("//third_party/mkl:intel_binary_blob"),
            ],
        ),
        linkstatic = True,
        **kwargs
    )
