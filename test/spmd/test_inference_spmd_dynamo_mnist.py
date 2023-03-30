import args_parse

SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'resnet50',
    },
    '--sharding': {
        'choices': ['batch', 'spatial', 'conv', 'linear'],
        'nargs': '+',
        'default': [],
    },
    '--use_virtual_device': {
        'action': 'store_true',
    },
    '--sample_count': {
        'type': int,
        'default': 10000,
    },
    '--test_set_batch_size': {
        'type': int,
        'default': 1,
    },
    '--target_latency': {
        'type': float,
        'default': 0.0005,
    },
    '--use_dynamo': {
        'action': 'store_true',
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
)

import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.test.test_utils as test_utils
import torch_xla.experimental.xla_sharding as xs
import torch_xla.experimental.pjrt as pjrt

DEFAULT_KWARGS = dict(
    target_latency=0.0005,  # 0.50 ms
)

# Set any args that were not explicitly given by the user.
default_value_dict = DEFAULT_KWARGS
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)



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


def inference_imagenet():
  print('==> Preparing data..')
  img_dim = 28
  if FLAGS.fake_data:
    assert FLAGS.test_set_batch_size == 1
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.test_set_batch_size, 1, img_dim, img_dim),
              torch.zeros(FLAGS.test_set_batch_size, dtype=torch.int64)),
        sample_count=FLAGS.sample_count // FLAGS.test_set_batch_size)
  else:
    # TODO real-data
    pass

  torch.manual_seed(42)

  device = xm.xla_device()
  model = MNIST().to(device)

  input_mesh = None
  if FLAGS.sharding:
    num_devices = pjrt.global_device_count()
    device_ids = np.arange(num_devices)
    # Model sharding
    if 'conv' in FLAGS.sharding:
      # Shard the model's convlution layers along two dimensions
      mesh_shape = (2, num_devices // 2, 1, 1)
      mesh = xs.Mesh(device_ids, mesh_shape, ('w', 'x', 'y', 'z'))
      partition_spec = (0, 1, 2, 3)  # Apply sharding along all axes
      print(
          f'Applying sharding to convolution layers with mesh {mesh.get_logical_mesh()}'
      )
      for name, layer in model.named_modules():
        if 'conv' in name:
          xs.mark_sharding(layer.weight, mesh, partition_spec)
    elif 'linear' in FLAGS.sharding:
      # Shard the model's fully connected layers across addressable devices
      mesh_shape = (num_devices, 1)
      mesh = xs.Mesh(device_ids, mesh_shape, ('x', 'y'))
      print(
          f'Applying sharding to linear layers with mesh {mesh.get_logical_mesh()}'
      )
      partition_spec = (0, 1)
      for name, layer in model.named_modules():
        if 'fc' in name:
          xs.mark_sharding(layer.weight, mesh, partition_spec)

    # Input sharding
    if 'batch' in FLAGS.sharding or 'spatial' in FLAGS.sharding:
      if 'batch' in FLAGS.sharding and 'spatial' in FLAGS.sharding:
        # Shard along both the batch dimension and spatial dimension
        # If there are more than 4 devices, shard along the height axis as well
        width_axis, height_axis = 2, num_devices // 4
        mesh_shape = (2, 1, width_axis, height_axis)
        input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
        print(
            f'Sharding input along batch and spatial dimensions with mesh {input_mesh.get_logical_mesh()}'
        )
      elif 'batch' in FLAGS.sharding:
        # Shard along batch dimension only
        mesh_shape = (num_devices, 1, 1, 1)
        input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
        print(
            f'Sharding input along batch dimension with mesh {input_mesh.get_logical_mesh()}'
        )
      elif 'spatial' in FLAGS.sharding:
        # Shard two-way along input spatial dimensions
        mesh_shape = (1, 1, num_devices // 2, 2)
        input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
        print(
            f'Sharding input images on spatial dimensions with mesh {input_mesh.get_logical_mesh()}'
        )
      test_loader = pl.MpDeviceLoader(
          test_loader,
          device,
          input_sharding=xs.ShardingSpec(input_mesh, (0, 1, 2, 3)))

  model.eval()
  xm.mark_step()
  xm.wait_device_ops()
  met.clear_all()

  if FLAGS.use_dynamo:
    print('Running torch.compile...')
    dynamo_model = torch.compile(model, backend='torchxla_trace_once')

  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(FLAGS.logdir)

  def inference_loop_fn(loader):
    start_0 = time.time()
    for step, (data, _) in enumerate(loader):
      if step == 1:
        start_1 = time.time()
      output = dynamo_model(data)
    end = time.time()
    return end - start_0, end-start_1

  latency, max_latency, avg_latency = 0.0, 0.0, 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    with torch.no_grad():
      elapsed_time_0, elapsed_time_1 = inference_loop_fn(test_loader)
    latency = elapsed_time_1
    avg_latency += latency
    xm.master_print(
      f'Elapsed time with example 0+: {elapsed_time_0}\n',
      f'Elapsed time with example 1+: {elapsed_time_1}')
    max_latency = max(latency, max_latency)

    if FLAGS.metrics_debug:
      xm.master_print(met.metrics_report())
  avg_latency /= FLAGS.sample_count

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Latency: {:.5f}%'.format(max_latency))
  xm.master_print('Avg Latency: {:.5f}%'.format(avg_latency))
  return avg_latency


if __name__ == '__main__':
  if FLAGS.profile:
    server = xp.start_server(FLAGS.profiler_port)

  torch.set_default_tensor_type('torch.FloatTensor')
  latency = inference_imagenet()
  if latency > FLAGS.target_latency:
    print('Latency {} is above target {}'.format(latency,
                                                  FLAGS.target_latency))
    sys.exit(21)
