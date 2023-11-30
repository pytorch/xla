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
from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
from torch_xla.experimental.xla_sharding import Mesh

import test_xla_sharding_base


class DebuggingSpmdTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    xr.use_spmd()
    super().setUpClass()

  @unittest.skipIf(
      not xr.using_pjrt() or
      xu.getenv_as(xenv.PJRT_DEVICE, str) in ("GPU", 'CUDA', 'ROCM', 'CPU'),
      f"Requires PJRT_DEVICE set to `TPU`.")
  def test_debugging_spmd_single_host_tiled(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = self.n_devices
    mesh_shape = (2, num_devices // 2)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)
    t = torch.randn(8, 4, device=device)
    partition_spec = (0, 1)
    xs.mark_sharding(t, mesh, partition_spec)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generated_table = visualize_tensor_sharding(t)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    # fake_console = rich.console.Console(file=io.StringIO(), width=120)
    color = None
    text_color = None
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU 0', "center", vertical="middle"),
            (2, 1, 2, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU 1', "center", vertical="middle"),
            (2, 1, 2, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU 2', "center", vertical="middle"),
            (2, 1, 2, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU 3', "center", vertical="middle"),
            (2, 1, 2, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU 4', "center", vertical="middle"),
            (2, 1, 2, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU 5', "center", vertical="middle"),
            (2, 1, 2, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU 6', "center", vertical="middle"),
            (2, 1, 2, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU 7', "center", vertical="middle"),
            (2, 1, 2, 1),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(
      not xr.using_pjrt() or
      xu.getenv_as(xenv.PJRT_DEVICE, str) in ("GPU", 'CUDA', 'ROCM', 'CPU'),
      f"Requires PJRT_DEVICE set to `TPU`.")
  def test_single_host_partial_replication(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = self.n_devices
    mesh_shape = (2, num_devices // 2)
    device_ids = np.array(range(num_devices))
    # mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
    mesh = self._get_mesh(mesh_shape)

    partition_spec = (0, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, (0, None))
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generated_table = visualize_tensor_sharding(t)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    fake_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU [0, 1, 2, 3]', "center", vertical="middle"),
            (2, 0, 2, 0),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('TPU [4, 5, 6, 7]', "center", vertical="middle"),
            (2, 0, 2, 0),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fake_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(
      not xr.using_pjrt() or
      xu.getenv_as(xenv.PJRT_DEVICE, str) in ("GPU", 'CUDA', 'ROCM', 'CPU'),
      f"Requires PJRT_DEVICE set to `TPU`.")
  def test_single_host_replicated(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = self.n_devices
    mesh_shape = (2, num_devices // 2)
    device_ids = np.array(range(num_devices))
    # mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
    mesh = self._get_mesh(mesh_shape)

    partition_spec_replicated = (None, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, partition_spec_replicated)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generated_table = visualize_tensor_sharding(t)
    console = rich.console.Console()
    with console.capture() as capture:
      console.print(generated_table)
    output = capture.get()

    color = None
    text_color = None
    fask_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align(
                'TPU [0, 1, 2, 3, 4, 5, 6, 7]', "center", vertical="middle"),
            (0, 0, 1, 0),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fask_table.add_row(*col)
    fake_console = rich.console.Console()
    with fake_console.capture() as fake_capture:
      fake_console.print(fake_table)
    fake_output = fake_capture.get()
    assert output == fake_output

  @unittest.skipIf(
      not xr.using_pjrt() or
      xu.getenv_as(xenv.PJRT_DEVICE, str) in ("GPU", 'CUDA', 'ROCM', 'TPU'),
      f"Requires PJRT_DEVICE set to `CPU`.")
  def test_debugging_spmd_single_host_tiled_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)
    t = torch.randn(8, 4, device=device)
    partition_spec = (0, 1)
    xs.mark_sharding(t, mesh, partition_spec)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generatedtable = visualize_tensor_sharding(t)
    console = rich.console.Console(file=io.StringIO(), width=120)
    console.print(generatedtable)
    output = console.file.getvalue()

    fake_console = rich.console.Console(file=io.StringIO(), width=120)
    color = None
    text_color = None
    fask_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [0]', "center", vertical="middle"),
            (0, 0, 1, 0),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fask_table.add_row(*col)
    fake_console.print(fask_table)
    fake_output = fake_console.file.getvalue()
    print("output: ")
    print(output.columns)
    print("fake_output: ")
    print(fake_output.columns)
    assert output == fake_output

  @unittest.skipIf(
      not xr.using_pjrt() or
      xu.getenv_as(xenv.PJRT_DEVICE, str) in ("GPU", 'CUDA', 'ROCM', 'TPU'),
      f"Requires PJRT_DEVICE set to `CPU`.")
  def test_single_host_partial_replication_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)

    partition_spec = (0, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, (0, None))
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generatedtable = visualize_tensor_sharding(t)
    console = rich.console.Console(file=io.StringIO(), width=120)
    console.print(generatedtable)
    output = console.file.getvalue()

    color = None
    text_color = None
    fask_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [0]', "center", vertical="middle"),
            (0, 5, 1, 4),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fask_table.add_row(*col)
    console.print(fask_table)
    fake_console = rich.console.Console(file=io.StringIO(), width=120)
    fake_console.print(fask_table)
    fake_output = fake_console.file.getvalue()
    print("output: ")
    print(output.columns)
    print("fake_output: ")
    print(fake_output.columns)
    assert output == fake_output

  @unittest.skipIf(
      not xr.using_pjrt() or
      xu.getenv_as(xenv.PJRT_DEVICE, str) in ("GPU", 'CUDA', 'ROCM', 'TPU'),
      f"Requires PJRT_DEVICE set to `CPU`.")
  def test_single_host_replicated_cpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)

    partition_spec_replicated = (None, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, partition_spec_replicated)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generatedtable = visualize_tensor_sharding(t)
    console = rich.console.Console(file=io.StringIO(), width=120)
    console.print(generatedtable)
    output = console.file.getvalue()

    color = None
    text_color = None
    fask_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('CPU [0]', "center", vertical="middle"),
            (0, 5, 1, 4),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fask_table.add_row(*col)
    fake_console = rich.console.Console(file=io.StringIO(), width=120)
    fake_console.print(fask_table)
    fake_output = fake_console.file.getvalue()
    print("output: ")
    print(output.columns)
    print("fake_output: ")
    print(fake_output.columns)
    assert output == fake_output

  @unittest.skipIf(not xr.using_pjrt() or
                   xu.getenv_as(xenv.PJRT_DEVICE, str) in ('CPU', 'TPU'),
                   f"Requires PJRT_DEVICE set to `GPU`, `CUDA`, `ROCM`.")
  def test_debugging_spmd_single_host_tiled_gpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)
    t = torch.randn(8, 4, device=device)
    partition_spec = (0, 1)
    xs.mark_sharding(t, mesh, partition_spec)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generatedtable = visualize_tensor_sharding(t)
    console = rich.console.Console(file=io.StringIO(), width=120)
    console.print(generatedtable)
    output = console.file.getvalue()

    fake_console = rich.console.Console(file=io.StringIO(), width=120)
    color = None
    text_color = None
    fask_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('GPU [0]', "center", vertical="middle"),
            (0, 0, 1, 0),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fask_table.add_row(*col)
    fake_console.print(fask_table)
    fake_output = fake_console.file.getvalue()
    assert output == fake_output

  @unittest.skipIf(not xr.using_pjrt() or
                   xu.getenv_as(xenv.PJRT_DEVICE, str) in ('CPU', 'TPU'),
                   f"Requires PJRT_DEVICE set to `GPU`, `CUDA`, `ROCM`.")
  def test_single_host_partial_replication_gpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)

    partition_spec = (0, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, (0, None))
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generatedtable = visualize_tensor_sharding(t)
    console = rich.console.Console(file=io.StringIO(), width=120)
    console.print(generatedtable)
    output = console.file.getvalue()

    color = None
    text_color = None
    fask_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('GPU [0]', "center", vertical="middle"),
            (0, 5, 1, 4),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fask_table.add_row(*col)
    console.print(fask_table)
    fake_console = rich.console.Console(file=io.StringIO(), width=120)
    fake_console.print(fask_table)
    fake_output = fake_console.file.getvalue()
    assert output == fake_output

  @unittest.skipIf(not xr.using_pjrt() or
                   xu.getenv_as(xenv.PJRT_DEVICE, str) in ('CPU', 'TPU'),
                   f"Requires PJRT_DEVICE set to `GPU`, `CUDA`, `ROCM`.")
  def test_single_host_replicated_gpu(self):
    from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
    device = xm.xla_device()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = self._get_mesh(mesh_shape)

    partition_spec_replicated = (None, None)
    t = torch.randn(8, 32, device=device)
    xs.mark_sharding(t, mesh, partition_spec_replicated)
    sharding = torch_xla._XLAC._get_xla_sharding_spec(t)
    generatedtable = visualize_tensor_sharding(t)
    console = rich.console.Console(file=io.StringIO(), width=120)
    console.print(generatedtable)
    output = console.file.getvalue()

    color = None
    text_color = None
    fask_table = rich.table.Table(
        show_header=False,
        show_lines=True,
        padding=0,
        highlight=True,
        pad_edge=False,
        box=rich.box.SQUARE)
    col = []
    col.append(
        rich.padding.Padding(
            rich.align.Align('GPU [0]', "center", vertical="middle"),
            (0, 5, 1, 4),
            style=rich.style.Style(bgcolor=color, color=text_color)))
    fask_table.add_row(*col)
    fake_console = rich.console.Console(file=io.StringIO(), width=120)
    fake_console.print(fask_table)
    fake_output = fake_console.file.getvalue()
    assert output == fake_output


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
