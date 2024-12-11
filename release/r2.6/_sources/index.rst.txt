:github_url: https://github.com/pytorch/xla

PyTorch/XLA documentation
===================================
PyTorch/XLA is a Python package that uses the XLA deep learning compiler to connect the PyTorch deep learning framework and Cloud TPUs.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Learn about Pytorch/XLA

   learn/xla-overview
   learn/pytorch-on-xla-devices
   learn/api-guide
   learn/dynamic_shape
   learn/eager
   learn/pjrt
   learn/troubleshoot

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Learn about accelerators

   accelerators/tpu
   accelerators/gpu

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Run ML workloads with Pytorch/XLA

   workloads/kubernetes

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: PyTorch/XLA features

   features/pallas.md
   features/stablehlo.md
   features/triton.md

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Improve Pytorch/XLA workload performance

   perf/amp
   perf/spmd_basic      
   perf/spmd_advanced
   perf/spmd_distributed_checkpoint
   perf/spmd_gpu
   perf/ddp
   perf/dynamo
   perf/fori_loop
   perf/fsdp
   perf/fsdpv2
   perf/quantized_ops
   perf/recompilation
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Contribute to Pytorch/XLA

   contribute/configure-environment
   contribute/codegen_migration
   contribute/op_lowering
   contribute/plugins
   contribute/bazel
   contribute/recompilation
