.. mdinclude:: ../../API_GUIDE.md

PyTorch/XLA API
==================================

xla_model
----------------------------------

.. automodule:: torch_xla.core.xla_model

.. autofunction:: xla_device
.. autofunction:: get_xla_supported_devices
.. autofunction:: xrt_world_size

.. autofunction:: get_ordinal
.. autofunction:: get_local_ordinal
.. autofunction:: is_master_ordinal

.. autofunction:: optimizer_step

.. autofunction:: save

distributed
----------------------------------

.. automodule:: torch_xla.distributed.parallel_loader

.. autoclass:: ParallelLoader
	       :members: per_device_loader

.. automodule:: torch_xla.distributed.data_parallel

.. autoclass:: DataParallel
	       :members: __call__

.. automodule:: torch_xla.distributed.xla_multiprocessing

.. autofunction:: spawn

utils
----------------------------------

.. automodule:: torch_xla.utils.utils

.. autoclass:: SampleGenerator

.. autoclass:: TfRecordReader
