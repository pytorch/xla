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
.. autofunction:: all_gather
.. autofunction:: all_to_all
.. autofunction:: add_step_closure
.. autofunction:: wait_device_ops
.. autofunction:: optimizer_step
.. autofunction:: save
.. autofunction:: rendezvous
.. autofunction:: do_on_ordinals
.. autofunction:: mesh_reduce
.. autofunction:: set_rng_state
.. autofunction:: get_rng_state
.. autofunction:: get_memory_info

.. automodule:: torch_xla.core.functions
.. autofunction:: all_reduce
.. autofunction:: all_gather
.. autofunction:: nms
		
distributed
----------------------------------

.. automodule:: torch_xla.distributed.parallel_loader
.. autoclass:: ParallelLoader
	       :members: per_device_loader

.. automodule:: torch_xla.distributed.xla_multiprocessing
.. autofunction:: spawn
.. autoclass:: MpModelWrapper
	       :members: to
.. autoclass:: MpSerialExecutor
	       :members: run

utils
----------------------------------

.. automodule:: torch_xla.utils.metrics
.. autofunction:: counter_names
.. autofunction:: counter_value
.. autofunction:: metric_names
.. autofunction:: metric_data
.. autofunction:: metrics_report
  
.. automodule:: torch_xla.utils.tf_record_reader
.. autoclass:: TfRecordReader

.. automodule:: torch_xla.utils.utils
.. autoclass:: SampleGenerator
.. autoclass:: DataWrapper

.. automodule:: torch_xla.utils.serialization
.. autofunction:: save
.. autofunction:: load

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
.. autofunction:: is_gcs_path

.. automodule:: torch_xla.utils.cached_dataset
.. autoclass:: CachedDataset


test
----------------------------------

.. automodule:: torch_xla.utils.test_utils
.. autofunction:: mp_test
.. autofunction:: write_to_summary
.. autofunction:: close_summary_writer
.. autofunction:: get_summary_writer
.. autofunction:: print_training_update
.. autofunction:: print_test_update

.. mdinclude:: ../../TROUBLESHOOTING.md
.. mdinclude:: ../pjrt.md
.. mdinclude:: ../ddp.md
.. mdinclude:: ../fsdp.md
.. mdinclude:: ../../OP_LOWERING_GUIDE.md
