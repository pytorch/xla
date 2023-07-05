import os
import torch
import argparse
import numpy as np
import datetime
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs
import torch_xla.debug.profiler as xp

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '--batch_size', type=int, default=131072, help="Per device Batch size.")
parser.add_argument('--embedding_dimension', type=int, default=2048)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--profile', action='store_true')
parser.add_argument(
    '--dcn_data_parallelism',
    type=int,
    default=1,
    help="N-way Data Parallelism across slices.")
parser.add_argument(
    '--dcn_fsdp_parallelism',
    type=int,
    default=1,
    help="Fsdp parallelism across slices that is expected to be 1 in most cases."
)
parser.add_argument(
    '--dcn_tensor_parallelism',
    type=int,
    default=1,
    help="Tensor parallelism across slices that is expected to be 1 in most cases."
)
parser.add_argument(
    '--ici_data_parallelism',
    type=int,
    default=1,
    help="Data parallelism within each slice that is expected to be 1 in most cases."
)
parser.add_argument(
    '--ici_fsdp_parallelism',
    type=int,
    default=4,
    help="Number of shards for Fsdp Parallelism within each slice.")
parser.add_argument(
    '--ici_tensor_parallelism',
    type=int,
    default=1,
    help="Number of shards for Tensor Parallelism within each slice.")
args = parser.parse_args()


def profile():
  from subprocess import Popen, PIPE
  import tempfile
  import sys
  import shutil
  tmpdir = tempfile.mkdtemp()
  cmd = f"/usr/local/bin/python /workspaces/work/pytorch/xla/scripts/capture_profile.py --service_addr 127.0.0.1:9012 --logdir {tmpdir} --duration_ms 1000"
  # cmd = f"/usr/bin/python /home/mohitkhatwani/pytorch/xla/scripts/capture_profile.py --service_addr 127.0.0.1:9012 --logdir {tmpdir} --duration_ms 10000"
  Popen(cmd.split()).communicate()
  label = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

  profile_dir = f'{tmpdir}/plugins/profile/'
  profiles = os.listdir(profile_dir)
  if len(profiles) != 1:
    print(
        "Warning: multiple profiles found after a single capture",
        profiles,
        file=sys.stderr)
  autoprof_dir = f'/tmp/home/autoprof/plugins/profile/'
  os.makedirs(autoprof_dir, exist_ok=True)
  for profile in profiles:
    shutil.move(f'{profile_dir}/{profile}', f'{autoprof_dir}/{label}_{profile}')
    cmd = f'/usr/bin/gsutil cp -r {autoprof_dir}/{label}_{profile} gs://jwtan/plugins/profile/{label}'
    # cmd = f'/snap/bin/gsutil cp -r {autoprof_dir}/{label}_{profile} gs://mohitkhatwani-pytorch-logs/plugins/profile/{label}'
    print(cmd)
    os.system(cmd)


# dcn_mesh_shape: shape of the logical mesh for the slower/outer network,
# in the same order as ici_parallelism.
dcn_parallelism = [
    args.dcn_data_parallelism, args.dcn_fsdp_parallelism,
    args.dcn_tensor_parallelism
]
# ici_parallelism: shape of the logical mesh for the faster/inner network, ordered
# by increasing network intensity, e.g. [data, fsdp, tensor] where tensor has
# the most network communication requirements.
ici_parallelism = [
    args.ici_data_parallelism, args.ici_fsdp_parallelism,
    args.ici_tensor_parallelism
]

num_devices = xr.global_device_count()
assert np.product(dcn_parallelism) * np.product(
    ici_parallelism) == num_devices, f"Number of devices {num_devices} \
    does not match the product of the parallelism {np.product(dcn_parallelism) * np.product(ici_parallelism)}"

device_ids = np.arange(num_devices)
# TODO: Change Mesh shape based on inputs of dcn_parallelism and ici_parallelism
ici_mesh_shape = (np.product(ici_parallelism), 1)
dcn_mesh_shape = (np.product(dcn_parallelism), 1)
data_mesh = xs.HybridMesh(
    ici_mesh_shape=ici_mesh_shape,
    dcn_mesh_shape=dcn_mesh_shape,
    axis_names=('data', 'tensor'))

mesh = xs.HybridMesh(
    ici_mesh_shape=ici_parallelism,
    dcn_mesh_shape=dcn_parallelism,
    axis_names=('data', 'fsdp', 'tensor'))

data_sharding = (0, 1)
parameter_sharding = (2, 1)


def gen_data(batch, d_emb):
  return torch.rand(batch, d_emb, dtype=torch.bfloat16, requires_grad=True)


def simple_timeit(f, tries=1, verbose=True):
  '''Simple utility to time a function for multiple runs'''
  s = datetime.datetime.now()
  f()
  e = datetime.datetime.now()
  outcomes = (e - s).total_seconds()
  return outcomes


class Layer(torch.nn.Module):

  def __init__(self, d_emb, d_ff):
    super(Layer, self).__init__()
    self.EMB2FF_linear = torch.nn.Linear(
        d_emb, d_ff, bias=False, dtype=torch.bfloat16)
    self.FF2EMB_linear = torch.nn.Linear(
        d_ff, d_emb, bias=False, dtype=torch.bfloat16)
    self.n1 = torch.nn.Sigmoid()
    self.n2 = torch.nn.Sigmoid()

  def forward(self, x):
    M1 = self.EMB2FF_linear(x)
    M1 = self.n1(M1)
    M2 = self.FF2EMB_linear(M1)
    M2 = self.n2(M2)
    return M2


class Model(torch.nn.Module):

  def __init__(self, num_layers):
    super(Model, self).__init__()
    self.layers = torch.nn.ModuleList([
        Layer(args.embedding_dimension, 4 * args.embedding_dimension)
        for _ in range(num_layers)
    ])

  def forward(self, x):
    for l in self.layers:
      x = l(x)
    return x


def my_loss(inputs):
  return torch.sum(inputs)


def training_step(data):
  model.train()
  with xp.Trace('train_sharding'):
    with xp.Trace('build_graph'):
      optimizer.zero_grad()
      output = model(data)
      loss = my_loss(output)
      loss.backward()
    optimizer.step()
  xm.mark_step()


torch.manual_seed(42)
tries = 5
device = xm.xla_device()

if args.profile:
  print("Profiler server started at port 9012")
  server = xp.start_server(9012)

global_batch_size = num_devices * args.batch_size
d_emb = args.embedding_dimension
d_ff = 4 * d_emb

import threading

data = gen_data(global_batch_size, args.embedding_dimension)
data = xm.send_cpu_data_to_device(data, device, input_sharding=xs.ShardingSpec(data_mesh, data_sharding))[0]

model = Model(args.num_layers).to(device)
for name, layer in model.named_modules():
  if 'linear' in name:
    xs.mark_sharding(layer.weight, mesh, parameter_sharding)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
parameters = 2 * d_ff * d_emb * args.num_layers
parameter_bytes = 2 * parameters
activation_bytes = 2 * (global_batch_size * (d_ff + d_emb)) * args.num_layers
memory_bytes = parameter_bytes + activation_bytes

print(
    f"total {memory_bytes/10**9} GB, parameters {parameter_bytes/10**9} GB, activations {activation_bytes/10**9} GB"
)
# N-parameter decoder-only model requires 6N matmul FLOPs per token seen.
# Detailed explaination in https://arxiv.org/pdf/2204.02311.pdf
TFLOPs_per_device = parameters * 6 * global_batch_size / 10**12 / xr.global_device_count(
)
outcomes = []
# warm up

# warmup tries
for _ in range(10):
  simple_timeit(lambda: training_step(data))

if args.profile:
  threading.Thread(target=profile).start()
for _ in range(tries):
  outcomes.append(simple_timeit(lambda: training_step(data)))
print(f"Timings {outcomes}")
time = sum(outcomes) / len(outcomes)

print(
    f"time is {time} seconds, TFLOP is {TFLOPs_per_device}, TFLOP/s is {TFLOPs_per_device/time}",
    flush=True)

# import torch_xla.debug.metrics as met
# print(met.metrics_report())
