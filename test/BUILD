load("//bazel:rules_def.bzl", "ptxla_py_test")

ptxla_py_test(
    name = "test_operations",
    srcs = ["test_operations.py"],
    accelerators = ["cpu"],
    args = ["--verbosity=2"],
    modes = [
        "default",
        "eager_debug",
        "dynamic",
        "without_functionalization",
    ],
)

ptxla_py_test(
    name = "test_async_closures",
    srcs = ["test_async_closures.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_xla_dist",
    srcs = ["test_xla_dist.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_profiler",
    srcs = ["test_profiler.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_ops",
    srcs = ["test_ops.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_metrics",
    srcs = ["test_metrics.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_zero1",
    srcs = ["test_zero1.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_data_type",
    srcs = ["test_data_type.py"],
    accelerators = ["cpu"],
    modes = [
        "downcast_bf16",
        "use_bf16",
    ],
)

ptxla_py_test(
    name = "test_env_var_mapper",
    srcs = ["test_env_var_mapper.py"],
    accelerators = ["cpu"],
    modes = [
        "xla_ir_debug",
        "xla_hlo_debug",
    ],
)

ptxla_py_test(
    name = "test_operations_hlo",
    srcs = ["test_operations_hlo.py"],
    accelerators = ["cpu"],
    args = ["--verbosity=2"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_input_output_aliases",
    srcs = ["test_input_output_aliases.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_torch_distributed_xla_backend",
    srcs = ["test_torch_distributed_xla_backend.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_replication",
    srcs = ["test_mp_replication.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_all_to_all",
    srcs = ["test_mp_all_to_all.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_collective_permute",
    srcs = ["test_mp_collective_permute.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_all_gather",
    srcs = ["test_mp_all_gather.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_reduce_scatter",
    srcs = ["test_mp_reduce_scatter.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_distributed_mm",
    srcs = ["test_mp_distributed_mm.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_save",
    srcs = ["test_mp_save.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_mesh_reduce",
    srcs = ["test_mp_mesh_reduce.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_mp_sync_batch_norm",
    srcs = ["test_mp_sync_batch_norm.py"],
    accelerators = ["cpu"],
    modes = [
        "default",
    ],
)

ptxla_py_test(
    name = "test_torch_distributed_all_gather_xla_backend",
    srcs = ["test_torch_distributed_all_gather_xla_backend.py"],
    accelerators = ["cpu"],
    modes = [
        "xla_backend_mp",
    ],
)

ptxla_py_test(
    name = "test_torch_distributed_all_reduce_xla_backend",
    srcs = ["test_torch_distributed_all_reduce_xla_backend.py"],
    accelerators = ["cpu"],
    modes = [
        "xla_backend_mp",
    ],
)

ptxla_py_test(
    name = "test_torch_distributed_multi_all_reduce_xla_backend",
    srcs = ["test_torch_distributed_multi_all_reduce_xla_backend.py"],
    accelerators = ["cpu"],
    modes = [
        "xla_backend_mp",
    ],
)

ptxla_py_test(
    name = "test_torch_distributed_reduce_scatter_xla_backend",
    srcs = ["test_torch_distributed_reduce_scatter_xla_backend.py"],
    accelerators = ["cpu"],
    modes = [
        "xla_backend_mp",
    ],
)

ptxla_py_test(
    name = "test_ddp",
    srcs = ["test_ddp.py"],
    accelerators = ["cpu"],
    modes = [
        "xla_backend_mp",
    ],
)

ptxla_py_test(
    name = "test_fsdp_auto_wrap",
    srcs = ["test_fsdp_auto_wrap.py"],
    accelerators = ["cpu"],
    modes = [
        "xla_backend_mp",
    ],
)

ptxla_py_test(
    name = "test_torch_distributed_fsdp_meta",
    srcs = ["test_torch_distributed_fsdp_meta.py"],
    accelerators = ["cpu"],
    modes = [
        "xla_backend_mp",
    ],
)

py_library(
    name = "test_utils",
    srcs = [
        "distributed_util.py",
        "test_utils.py",
    ],
)
