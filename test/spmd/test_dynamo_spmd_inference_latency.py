"""
DO NOT MERGE - this is an experiment script.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as met
import torch.optim as optim
import torchvision
import unittest
import numpy as np

import torch_xla.experimental.xla_sharding as xs
import torch_xla.distributed.parallel_loader as pl
import torch_xla.experimental.pjrt as pjrt

class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


for b in [1, 128, 512]:
    print(f'batch size {b}')
    for spatial in [False, True]:
        print(f'spatial: {spatial}')
        for conv in [False, True]:
            print(f'conv: {conv}')
            device = xm.xla_device()
            xla_resnet50 = MNIST()  #torchvision.models.resnet50()
            xla_resnet50.to(device)
            xla_resnet50.eval()

            batch_size = b #xu.getenv_as('BATCH_SIZE', int, defval=128)
            sample_count = xu.getenv_as('SAMPLE_COUNT', int, defval=1000)
            # 3, 224 vs 1, 28
            loader = xu.SampleGenerator(
                data=(torch.randn(batch_size, 1, 28, 28, device=device),
                      torch.zeros(batch_size, dtype=torch.int64, device=device)),
                sample_count=sample_count)

            # For ResNet50 is not wide/deep enough to justify the conv layer sharding.
            # Spatial sharding yields the best performance, when batch size is large enough.
            if spatial:
                num_devices = pjrt.global_device_count()
                device_ids = np.arange(num_devices)
                mesh_shape = (1, 1, num_devices // 2, 2)
                input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
                loader = pl.MpDeviceLoader(
                          loader,
                          device,
                          input_sharding=xs.ShardingSpec(input_mesh, (0, 1, 2, 3)))
                if conv:
                    mesh_shape = (2, num_devices // 2, 1, 1)
                    mesh = xs.Mesh(device_ids, mesh_shape, ('w', 'x', 'y', 'z'))
                    partition_spec = (0, 1, 2, 3)  # Apply sharding along all axes
                    for name, layer in xla_resnet50.named_modules():
                      if 'conv' in name:
                        xs.mark_sharding(layer.weight, mesh, partition_spec)
            else:
                loader = pl.MpDeviceLoader(loader, device)

            # materalize the fake data for test purpose
            xm.mark_step()
            xm.wait_device_ops()
            met.clear_all()
            dynamo_resnet50 = torch.compile(
              xla_resnet50, backend='torchxla_trace_once')

            import time
            avg0, max0, avg1, max1 = 0., 0., 0., 0.
            cnt = 0
            for step, (data, _) in enumerate(loader):
              start = time.time()
              output = dynamo_resnet50(data)
              end = time.time()
              lat = end - start
              avg0 += lat
              if step != 0:
                  avg1 += lat
                  max1 = max(max1, lat)
              else:
                  max0 = lat
              cnt += 1
            avg0 /= cnt
            avg1 /= (cnt - 1)

            # We only expect one graph for the resnet18 inference.
            if met.metric_data('CompileTime'):
                print('compile time: ',met.metric_data('CompileTime')[0])
            print('latency 0: ', avg0 * 1000, max0 * 1000)
            print('latency 1: ', avg1 * 1000, max1 * 1000)
