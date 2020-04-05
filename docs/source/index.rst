.. mdinclude:: ../../API_GUIDE.md

PyTorch/XLA API
==================================

xla_model
----------------------------------

.. automodule:: torch_xla.core.xla_model
.. autofunction:: xla_device
.. autofunction:: get_xla_supported_devices
.. autofunction:: xla_device_hw
.. autofunction:: get_ordinal
.. autofunction:: get_local_ordinal
.. autofunction:: is_master_ordinal
.. autofunction:: xrt_world_size
.. autofunction:: all_reduce
.. autofunction:: all_to_all
.. autofunction:: add_step_closure
.. autofunction:: wait_device_ops
.. autofunction:: optimizer_step
.. autofunction:: save
.. autofunction:: rendezvous
.. autofunction:: mesh_reduce

distributed
----------------------------------

.. automodule:: torch_xla.distributed.parallel_loader
.. autoclass:: ParallelLoader
	       :members: per_device_loader

.. automodule:: torch_xla.distributed.xla_multiprocessing
.. autofunction:: spawn

utils
----------------------------------

.. automodule:: torch_xla.utils.tf_record_reader
.. autoclass:: TfRecordReader

.. automodule:: torch_xla.utils.utils
.. autoclass:: SampleGenerator
.. autoclass:: DataWrapper

.. automodule:: torch_xla.utils.gcsfs
.. autofunction:: open
.. autofunction:: list
.. autofunction:: stat
.. autofunction:: remove
.. autofunction:: rmtree
.. autofunction:: read
.. autofunction:: write
.. autofunction:: generic_open
.. autofunction:: generic_read
.. autofunction:: generic_write


