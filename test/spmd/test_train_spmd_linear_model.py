import args_parse
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch_xla.distributed.xla_backend
from torchdata.datapipes.iter import IterableWrapper
from torchdata.dataloader2 import DistributedReadingService, DataLoader2
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.utils.checkpoint as checkpoint
import torch_xla.utils.utils as xu
from torch_xla.distributed.spmd import Mesh
import torch.optim as optim
from torch import nn

MODEL_OPTS = {
    '--sharding': {
        'choices': ['batch', 'megatron-lm', 'fsdp'],
        'nargs': '+',
        'default': [],
    },
    '--input_dim': {
        'type': int,
        'default': 784,
    },
    '--train_dataset_len': {
        'type': int,
        'default': 1024 * 1024,
    },
    '--use_gradient_checkpointing': {
        'action': 'store_true',
    },
    '--persistent_workers': {
        'action': 'store_true',
    },
    '--prefetch_factor': {
        'type': int,
    },
    '--loader_prefetch_size': {
        'type': int,
    },
    '--load_from_chkpt': {
        'type': bool,
        'default': False
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




def transfer_to_device(data, target, mesh):
  from_cpu_shards = torch_xla._XLAC._global_tensor_from_cpu_shards
  op_sharding = mesh.get_op_sharding((0, 1))
  local_ndev = len(torch_xla._XLAC._xla_get_runtime_devices())
  data_shards_cpu = list(torch.split(data, data.shape[0]//local_ndev))
  target_shards_cpu = list(torch.split(target, target.shape[0]//local_ndev))
  data = from_cpu_shards(data_shards_cpu, op_sharding)
  target_op_sharding = mesh.get_op_sharding((0,))
  target = from_cpu_shards(target_shards_cpu, target_op_sharding) # TODO: ask this
  return data, target


def train():
  print('===> Preparing data..')
  lr = 0.001
  if FLAGS.fake_data:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, FLAGS.input_dim),
              torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
        sample_count=FLAGS.train_dataset_len // FLAGS.batch_size)
  else:
    torch.distributed.init_process_group('gloo', init_method='xla://')
    trainset = torchvision.datasets.MNIST(root='/tmp/data', train=True, download=True, transform=transforms.ToTensor())
    trainset = IterableWrapper(trainset).sharding_filter().batch(batch_size = FLAGS.batch_size // xr.process_count(), drop_last = FLAGS.drop_last)
    rs = DistributedReadingService()
    train_loader = DataLoader2(trainset, reading_service = rs)
    # train_loader = torch.utils.data.DataLoader(
    #   trainset, 
    #   batch_size=FLAGS.batch_size,
    #   drop_last=FLAGS.drop_last,
    #   shuffle=True,
    #   num_workers=FLAGS.num_workers,
    #   persistent_workers=FLAGS.persistent_workers,
    #   prefetch_factor=FLAGS.prefetch_factor
    # )

  torch.manual_seed(42)
  model = SimpleLinear().to(device)

  num_devices = xr.global_runtime_device_count()
  print(f'num_devices: {num_devices}')
  # Define a mesh with all devices along one axis
  mesh_shape = (num_devices, 1)
  device_ids = np.arange(num_devices)
  mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

  # if 'batch' in FLAGS.sharding:
  #   train_loader = pl.MpDeviceLoader(
  #       train_loader, device, input_sharding=xs.ShardingSpec(mesh, (0, 1)))

  if 'fsdp' in FLAGS.sharding:
    # train_loader = pl.MpDeviceLoader(
    #     train_loader, device, input_sharding=xs.ShardingSpec(mesh, (0, 1)))
    print('Sharding model weights')
    # Shard the weights according to their 0th dim
    xs.mark_sharding(model.fc1.weight, mesh, (0, 1))
    xs.mark_sharding(model.fc2.weight, mesh, (0, 1))

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
    for step, chunk in enumerate(loader):
      data = torch.empty((0,) + chunk[0][0].shape[1:], dtype = chunk[0][0].dtype)
      target = torch.empty((0,))
      for d, t in chunk:
        data = torch.cat((data, d))
        target = torch.cat((target, torch.tensor([t])))
      if not FLAGS.fake_data:
        data_shape = data.shape
        data = torch.reshape(data, (data_shape[0], data_shape[-1] * data_shape[-2]))
      if step == 100: # save state at step 100
        torch.save({'dataloader': loader.state_dict(), 'step': step}, '/tmp/chkpt')
      with xp.StepTrace('train_linear_model'):
        with xp.Trace('build_graph'):
          x, target = transfer_to_device(data, target, mesh)
          y = target.to(device)
          optimizer.zero_grad()
          if FLAGS.use_gradient_checkpointing:
            for n_l, layer in enumerate(model):
              # Apply gradient checkpointing for reduced memory footprint.
              # This would result in increased computation cost.
              if n_l > 0:
                x = torch_xla.utils.checkpoint.checkpoint(layer, x)
            output = x
          else:
            output = model(x)
          loss = loss_fn(output, y)
          loss.backward()
        optimizer.step()
      xm.mark_step()
      if step % 10 == 0:
        print(f"Epoch {epoch} step {step} loss {loss}")

  for epoch in range(FLAGS.num_epochs):
    if FLAGS.load_from_chkpt:
      state_dict = torch.load('/tmp/chkpt')
      train_loader.load_state_dict(state_dict['dataloader'])
    train_loop_fn(train_loader, epoch)

  return model


if FLAGS.profile:
  server = xp.start_server(FLAGS.profiler_port)

print('Start training loop...')
torch_xla.runtime.use_spmd()
device = xm.xla_device()
m = train()
t = torch.randn(10, FLAGS.input_dim).to(device)
m(t).cpu()
