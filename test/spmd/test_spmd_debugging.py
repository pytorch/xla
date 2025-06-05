import sys

import unittest
from unittest.mock import patch
import math
import numpy as np
import os
import io
import rich

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import XLAShardedTensor
from torch_xla.distributed.spmd import Mesh

import test_xla_sharding_base


class DebuggingSpmdTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  @unittest.skipIf(
      xr.device_type() == 'CPU',
      f"Requires PJRT_DEVICE set to `TPU`, `GPU`, `CUDA`, or 'ROCM'.")
  def test_debugging_spmd_single_host_tiled_tpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{devices=[2,4]0,1,2,3,4,5,6,7}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 0', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 1', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 2', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 3', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 4', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 5', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 6', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 7', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(
      xr.device_type() == 'CPU',
      f"Requires PJRT_DEVICE set to  `TPU`, `GPU`, `CUDA`, or 'ROCM'.")
  def test_single_host_partial_replication_tpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{devices=[4,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [0, 1]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [2, 3]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [4, 5]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [6, 7]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(
      xr.device_type() == 'CPU',
      f"Requires PJRT_DEVICE set to `TPU`, `GPU`, `CUDA`, or 'ROCM'.")
  def test_single_host_replicated_tpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{replicated}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    alltpus = xr.device_type() + ' ' + str(
        list(range(xr.global_runtime_device_count())))
    col.append(
        rich.padding.Padding(
            rich.align.Align(alltpus, "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(xr.device_type() != 'CPU',
                   f"Requires PJRT_DEVICE set to `CPU`.")
  def test_debugging_spmd_single_host_tiled_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    device = torch_xla.device()
    num_devices = self.n_devices
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)

    partition_spec = (0, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, (0, None))
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [0]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(xr.device_type() != 'CPU',
                   f"Requires PJRT_DEVICE set to `CPU`.")
  def test_single_host_partial_replication_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    device = torch_xla.device()
    num_devices = self.n_devices
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)

    partition_spec = (0, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, (0, None))
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [0]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(xr.device_type() != 'CPU',
                   f"Requires PJRT_DEVICE set to `CPU`.")
  def test_single_host_replicated_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    device = torch_xla.device()
    num_devices = self.n_devices
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)

    partition_spec_replicated = (None, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, partition_spec_replicated)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [0]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output


# Multi-host tests
# e.g.: sharding={devices=[2,8]0,4,8,12,2,6,10,14,1,5,9,13,3,7,11,15}
# e.g.: sharding={devices=[8,1,2]0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15 last_tile_dim_replicate}
# e.g.: sharding={replicated}

  @unittest.skipIf(
      xr.device_type() == 'CPU',
      f"Requires PJRT_DEVICE set to `TPU`, `GPU`, `CUDA`, or 'ROCM'.")
  def test_debugging_spmd_multi_host_tiled_tpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{devices=[2,8]0,4,8,12,2,6,10,14,1,5,9,13,3,7,11,15}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 0', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 4', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 8', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 12', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 2', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 6', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 10', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 14', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 1', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 5', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 9', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 13', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 3', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 7', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 11', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' 15', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(
      xr.device_type() == 'CPU',
      f"Requires PJRT_DEVICE set to `TPU`, `GPU`, `CUDA`, or 'ROCM'.")
  def test_multi_host_partial_replication_tpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{devices=[8,1,2]0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15 last_tile_dim_replicate}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [0, 1]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [4, 5]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [8, 9]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [12, 13]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [2, 3]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [6, 7]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [10, 11]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [14, 15]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(
      xr.device_type() == 'CPU',
      f"Requires PJRT_DEVICE set to `TPU`, `GPU`, `CUDA`, or 'ROCM'.")
  @unittest.skipIf(xr.global_runtime_device_count() != 8,
                   f"Limit test num_devices to 8 for function consistency")
  def test_multi_host_replicated_tpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{replicated}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                xr.device_type() + ' [0, 1, 2, 3, 4, 5, 6, 7]',
                "center",
                vertical="middle"), (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(xr.device_type() != 'CPU',
                   f"Requires PJRT_DEVICE set to `CPU`.")
  def test_debugging_spmd_multi_host_tiled_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{devices=[2,8]0,4,8,12,2,6,10,14,1,5,9,13,3,7,11,15}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 0', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 4', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 8', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 12', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 2', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 6', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 10', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 14', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 1', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 5', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 9', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 13', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 3', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 7', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 11', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU 15', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(xr.device_type() != 'CPU',
                   f"Requires PJRT_DEVICE set to `CPU`.")
  def test_multi_host_partial_replication_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{devices=[8,1,2]0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15 last_tile_dim_replicate}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [0, 1]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [4, 5]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [8, 9]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [12, 13]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [2, 3]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [6, 7]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [10, 11]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [14, 15]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(xr.device_type() != 'CPU',
                   f"Requires PJRT_DEVICE set to `CPU`.")
  def test_multi_host_replicated_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_sharding
    sharding = '{replicated}'
    generated_table = visualize_sharding(sharding)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    use_color = True if rich.console.Console().color_system else False
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None)
    col = []
    # PJRT_DEVICE=CPU will only has one CPU, please update once situation change
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [0]', "center", vertical="middle"),
            (1, 1, 1, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
