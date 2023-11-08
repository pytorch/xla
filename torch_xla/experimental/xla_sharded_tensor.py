# Keep this for backward compatibility.
# TODO(yeounoh) remove after 2.2 release.
import warnings

warnings.warn(
    "Importing from `torch_xla.experimental.xla_sharded_tensor` will be deprecated "
    "after 2.2 release. Please use `torch_xla.experimental.spmd` "
    "instead.", DeprecationWarning, 2)

from .spmd.xla_sharded_tensor import *
