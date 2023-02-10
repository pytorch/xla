import args_parse
import numpy as np
import torch
from torch import nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.experimental.xla_sharding as xs
import torch_xla.utils.utils as xu
from torch_xla.experimental.xla_sharding import Mesh
import torch.optim as optim
from torch import nn

MODEL_OPTS = {
    '--sharding': {
        'choices': ['batch', 'megatron-lm'],
        'nargs': '+',
        'default': [],
    },
    '--input_dim': {
        'type': int,
        'default': 16834,
    },
    '--train_dataset_len': {
        'type': int,
        'default': 1024 * 1024,
    },
}

FLAGS = args_parse.parse_common_options(
    batch_size=128, num_epochs=1, opts=MODEL_OPTS.items())


class SimpleLinear(nn.Module):

  def __init__(self):
    super(SimpleLinear, self).__init__()
    self.fc1 = nn.Linear(FLAGS.input_dim, FLAGS.input_dim // 2)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(FLAGS.input_dim // 2, 1)
    # Add an additional 1x1 layer at the end to ensure the final layer
    # is not sharded.
    self.fc3 = nn.Linear(1, 1)

  def forward(self, x):
    y = self.relu(self.fc1(x))
    z = self.fc2(y)
    return self.fc3(z)


device = xm.xla_device()


def train():
  print('===> Preparing data..')
  num_epochs = 18
  lr = 0.1
  train_loader = xu.SampleGenerator(
      data=(torch.zeros(FLAGS.batch_size, FLAGS.input_dim),
            torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
      sample_count=FLAGS.train_dataset_len // FLAGS.batch_size)
  torch.manual_seed(42)
  model = SimpleLinear().to(device)

  num_devices = len(xm.get_xla_supported_devices())
  print(f'num_devices: {num_devices}')
  # Define a mesh with all devices along one axis
  mesh_shape = (num_devices, 1)
  device_ids = np.arange(num_devices)
  mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

  if 'megatron-lm' in FLAGS.sharding:
    print('Sharding model weights')
    # Shard the first layer's weights row-wise
    xs.mark_sharding(model.fc1.weight, mesh, (0, 1))
    # Shard the second layer's weights column-wise
    xs.mark_sharding(model.fc2.weight, mesh, (1, 0))

  optimizer = optim.SGD(model.parameters(), lr=lr)

  loss_fn = nn.CrossEntropyLoss()

  def train_loop_fn(loader, epoch):
    model.train()
    for step, (data, target) in enumerate(loader):
      with xp.StepTrace('train_linear_model'):
        with xp.Trace('build_graph'):
          data = data.to(device)
          target = target.to(device)
          if 'batch' in FLAGS.sharding:
            # All devices are along axis 0, which corresponds to the
            # batch axis for the input
            xs.mark_sharding(data, mesh, (0, 1))
          optimizer.zero_grad()
          output = model(data)
          loss = loss_fn(output, target)
          loss.backward()
        optimizer.step()
      xm.mark_step()
      if step % 10 == 0:
        print(f"Epoch {epoch} step {step} loss {loss}")

  for epoch in range(FLAGS.num_epochs):
    train_loop_fn(train_loader, epoch)

  return model


if FLAGS.profile:
  server = xp.start_server(FLAGS.profiler_port)

print('Start training loop...')
m = train()
t = torch.randn(10, FLAGS.input_dim).to(device)
m(t).cpu()
