torch_xla
===================================

PyTorch runs on TPUs with the `torch_xla`_ package.

XLA devices
----------------------------------

The XLA device type defines virtual 'xla' devices backed by real hardware.
'xla:0' is your machine's CPU and is useful for testing.
'xla:1' and above represent Cloud TPUs.

XLA devices work like CUDA devices. Operations on an XLA tensor occur on its
device and cross-device operations (other than copy) are not allowed.

Lazy execution
----------------------------------

PyTorch usually runs operations immediately or 'eagerly.' XLA devices, however,
executed lazily. Operations on XLA tensors are usually recorded in a graph,
not executed, until a tensor's values are needed. Deferring execution lets XLA
reduce roundtrips to Cloud TPUs and optimize computation by reviewing multiple
operations simultaneously. A series of operations might be fused into a single
operation, for example.

bFloat16 and fp32
----------------------------------

XLA devices can treat the 'float' datatype as `bfloat16`_ by setting the
'XLA_USE_BF16' environment variable. If set to 1, XLA 'float' tensors actually
use bfloat16.

On Cloud TPUs the 'double' datatype is treated as 'float,' so when
'XLA_USE_BF16=1' the Cloud TPUs map 'double' to 'float' and 'float' to 'bfloat16.'

Contiguity and storage
----------------------------------

XLA tensors are always contiguous and, unlike CPU and CUDA tensors, have no
notion of storage. Models should not assume any particular striding for XLA
tensors.

TorchXLA classes and functions
----------------------------------

TorchXLA defines some new classes and functions for managing XLA devices,
performing distributed training, and taking optimizer steps. See its
`API Guide`_ for examples of how to use this new functionality and run models
on Cloud TPUs. The documentation for each of these classes and functions is
below.

.. _torch_xla: https://github.com/pytorch/xla
.. _bfloat16: https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
.. _API Guide: https://github.com/pytorch/xla/blob/master/API_GUIDE.md

xla_model
----------------------------------

.. automodule:: torch_xla.core.xla_model

.. autofunction:: is_xla_tensor
.. autofunction:: get_xla_supported_devices
.. autofunction:: xla_device
.. autofunction:: xla_real_devices
.. autofunction:: xla_replication_devices
.. autofunction:: set_replication

.. autofunction:: xrt_world_size

.. autofunction:: get_ordinal
.. autofunction:: is_master_ordinal
.. autofunction:: master_print

.. autofunction:: optimizer_step

distributed
----------------------------------

.. automodule:: torch_xla.distributed.parallel_loader
   :members:

.. automodule:: torch_xla.distributed.data_parallel
   :members:

.. automodule:: torch_xla.distributed.xla_multiprocessing
   :members:

utils
----------------------------------

.. automodule:: torch_xla.utils.utils
   :members:
