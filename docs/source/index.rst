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
.. autofunction:: is_master_ordinal

.. autofunction:: optimizer_step

distributed
----------------------------------

.. automodule:: torch_xla.distributed.parallel_loader

.. autoclass:: ParallelLoader

.. automodule:: torch_xla.distributed.data_parallel

.. autoclass:: DataParallel

.. automodule:: torch_xla.distributed.xla_multiprocessing

.. autofunction:: spawn

utils
----------------------------------

.. automodule:: torch_xla.utils.utils


.. autoclass:: SampleGenerator