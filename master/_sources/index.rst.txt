:github_url: https://github.com/pytorch/xla

PyTorch/XLA documentation
===================================
``torch_xla`` is a Python package that implements \
`XLA <https://openxla.org/xla>`_ as a backend for PyTorch.

+------------------------------------------------+------------------------------------------------+------------------------------------------------+
| **Familiar APIs**                              | **High Performance**                           | **Cost Efficient**                             |
|                                                |                                                |                                                |
| Create and train PyTorch models on TPUs,       | Scale training jobs across thousands of        | TPU hardware and the XLA compiler are optimized|
| with only minimal changes required.            | TPU cores while maintaining high MFU.          | for cost-efficient training and inference.     |
+------------------------------------------------+------------------------------------------------+------------------------------------------------+

Getting Started
---------------

Install with pip.

.. code-block:: sh

   pip install torch torch_xla[tpu]

Verify the installation:

.. code-block:: sh

   python -c "import torch_xla; print(torch_xla.__version__)"
   python -c "import torch; import torch_xla; print(torch.tensor(1.0, device='xla').device)"

Tutorials
---------

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Learn the Basics

   learn/pytorch-on-xla-devices
   learn/xla-overview

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Distributed Training on TPU

   accelerators/tpu
   tutorials/precision_tutorial
   perf/spmd_basic      
   perf/spmd_advanced
   perf/spmd_distributed_checkpoint
   features/torch_distributed
   perf/ddp
   perf/fsdp_collectives
   perf/fsdp_spmd

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced Techniques

   features/pallas
   features/stablehlo
   perf/amp
   learn/dynamic_shape
   perf/dynamo
   perf/quantized_ops
   features/scan
   perf/fori_loop
   perf/assume_pure

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Troubleshooting

   learn/troubleshoot
   learn/eager
   notes/source_of_recompilation
   perf/recompilation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Training on GPU

   accelerators/gpu
   features/triton
   perf/spmd_gpu

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Contributing

   contribute/bazel
   contribute/configure-environment
   contribute/cpp_debugger
   contribute/op_lowering
   contribute/codegen_migration
   contribute/plugins

API Reference
-------------

.. toctree::
   :glob:
   :maxdepth: 2

   learn/api-guide
