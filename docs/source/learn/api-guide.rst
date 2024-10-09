
PyTorch/XLA API
==================================

torch_xla
----------------------------------
.. automodule:: torch_xla
.. autofunction:: device
.. autofunction:: devices
.. autofunction:: device_count
.. autofunction:: sync
.. autofunction:: compile
.. autofunction:: manual_seed

runtime
----------------------------------
.. automodule:: torch_xla.runtime
.. autofunction:: device_type
.. autofunction:: local_process_count
.. autofunction:: local_device_count
.. autofunction:: addressable_device_count
.. autofunction:: global_device_count
.. autofunction:: global_runtime_device_count
.. autofunction:: world_size
.. autofunction:: global_ordinal
.. autofunction:: local_ordinal
.. autofunction:: get_master_ip
.. autofunction:: use_spmd
.. autofunction:: is_spmd
.. autofunction:: initialize_cache


xla_model
----------------------------------

.. automodule:: torch_xla.core.xla_model
.. autofunction:: xla_device
.. autofunction:: xla_device_hw
.. autofunction:: is_master_ordinal
.. autofunction:: all_reduce
.. autofunction:: all_gather
.. autofunction:: all_to_all
.. autofunction:: add_step_closure
.. autofunction:: wait_device_ops
.. autofunction:: optimizer_step
.. autofunction:: save
.. autofunction:: rendezvous
.. autofunction:: mesh_reduce
.. autofunction:: set_rng_state
.. autofunction:: get_rng_state
.. autofunction:: get_memory_info
.. autofunction:: get_stablehlo
.. autofunction:: get_stablehlo_bytecode

distributed
----------------------------------

.. automodule:: torch_xla.distributed.parallel_loader
.. autoclass:: MpDeviceLoader

.. automodule:: torch_xla.distributed.xla_multiprocessing
.. autofunction:: spawn

spmd
----------------------------------
.. automodule:: torch_xla.distributed.spmd
.. autofunction:: mark_sharding
.. autofunction:: clear_sharding
.. autofunction:: set_global_mesh
.. autofunction:: get_global_mesh
.. autofunction:: get_1d_mesh
.. autoclass:: Mesh
.. autoclass:: HybridMesh

experimental
----------------------------------
.. automodule:: torch_xla.experimental
.. autofunction:: eager_mode

debug
----------------------------------

.. automodule:: torch_xla.debug.metrics
.. autofunction:: metrics_report
.. autofunction:: short_metrics_report
.. autofunction:: counter_names
.. autofunction:: counter_value
.. autofunction:: metric_names
.. autofunction:: metric_data