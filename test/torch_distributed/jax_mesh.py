import jax
from jax._src import xla_bridge as xb
from jax._src import mesh as mesh_lib
import numpy as np

mesh = jax.make_mesh((jax.device_count(), ), ('x', ))
print(mesh.device_ids)

devices = xb.devices()
new_mesh_devices = np.asarray([devices[0], devices[2], devices[4], devices[6], devices[7], devices[5], devices[3], devices[1]])
new_mesh = mesh_lib.Mesh(new_mesh_devices, "x", axis_types=None)
print(new_mesh.device_ids)