from dataclasses import dataclass

@dataclass
class SPMDConfig:
  """
  PyTorch/XLA SPMD configurations. These are optional configuration settings,
  passed to use_spmd API

  ```
  import torch_xla.runtime as xr
  import torch_xla.distributed.spmd.AutoSPMDConfig

  xr.use_spmd(auto=True, spmd_config = SPMDConfig())
  ```
  """


class AutoSPMDConfig(SPMDConfig):
  """
  PyTorch/XLA SPMD configurations for auto-sharding.

  ```
  import torch_xla.runtime as xr
  import torch_xla.distributed.spmd.AutoSPMDConfig

  configs = {"keep_user_sharding" : True}
  xr.use_spmd(auto=True, spmd_config = AutoSPMDConfig(**configs))
  ```
  """
  auto_partitioner: str = "alpa"  # default: use alpa partitioning algorithm
  keep_user_sharding: bool = False  # default: keeps input/output shardings
  memory_budget: int  # default: do not specify the budget.
  mesh_shape: str  # default: if unset, auto-sharding uses
  auto_mesh_selection: bool = False  # default: do not explore mesh automatically.



