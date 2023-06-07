"""Rules that simplify deps and compiler configuration for PyTorch/XLA."""

load(
    "@tsl//tsl/platform/default:rules_cc.bzl",
    "cc_test",
)

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

def ptxla_cc_test(
        deps,
        copts = [],
        **kwargs):
    cc_test(
        name = "%s%s" % (name, ""),
        srcs = srcs + tf_binary_additional_srcs(),
        copts = tf_copts() + copts + [
            "-isystemexternal/torch",  # Required for system includes.
            "-fexceptions",  # Required for testing crashes.
        ],
        linkopts = select({
            "//conditions:default": [
                "-lpthread",
                "-lm",
            ],
            clean_dep("//third_party/compute_library:build_with_acl"): [
                "-fopenmp",
                "-lm",
            ],
        }) + lrt_if_needed() + _rpath_linkopts(name),
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
        data = [] +
               tf_binary_dynamic_kernel_dsos() +
               tf_binary_additional_srcs(),
        exec_properties = tf_exec_properties(kwargs),
        linkstatic = True,
        **kwargs
    )
