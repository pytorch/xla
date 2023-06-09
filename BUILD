load("@tsl//tsl/platform/default:rules_cc.bzl", "cc_binary")

cc_binary(
    name = "_XLAC.so",
    linkopts = [
        "-Wl,-rpath,$$ORIGIN/torch_xla/lib",  # for libtpu
        "-Wl,-soname,_XLAC.so",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//torch_xla/csrc:init_python_bindings",
        "//torch_xla/csrc:tensor",
        "//torch_xla/csrc:version",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@xla//xla/python/profiler/internal:traceme_wrapper",
        "@xla//xla/service:hlo_parser",
        "@xla//xla/service:hlo_pass_pipeline",
        "@xla//xla/service:hlo_verifier",
        "@xla//xla/service:hlo_proto_util",
        "@xla//xla/service:sharding_propagation",
        "@xla//xla/service/spmd:spmd_partitioner",
        "@tsl//tsl/profiler/lib:traceme",
        "@torch//:headers",
        "@torch//:libc10",
        "@torch//:libtorch",
        "@torch//:libtorch_cpu",
        "@torch//:libtorch_python",
    ],
)
