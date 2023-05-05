import os
import torch
import argparse
import numpy as np
import datetime
import torch_xla.core.xla_model as xm
import torch_xla.experimental.pjrt as pjrt
import torch_xla.experimental.xla_sharding as xs
import torch_xla.debug.profiler as xp

os.environ["XLA_USE_SPMD"] = "1"
os.environ["PJRT_DEVICE"] = "TPU"

devices = xm.get_xla_supported_devices(devkind='TPU')

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--batch_size', type=int, default=131072)
parser.add_argument('--embedding_dimension', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--profile', type=bool, default=False)
parser.add_argument('--dcn_data_parallelism', type=int, default=1)
parser.add_argument('--dcn_fsdp_parallelism', type=int, default=1)
parser.add_argument('--dcn_tensor_parallelism', type=int, default=1)
parser.add_argument('--ici_data_parallelism', type=int, default=1)
parser.add_argument('--ici_fsdp_parallelism', type=int, default=4)
parser.add_argument('--ici_tensor_parallelism', type=int, default=1)
args = parser.parse_args()

dcn_parallelism = [
    args.dcn_data_parallelism, args.dcn_fsdp_parallelism,
    args.dcn_tensor_parallelism
]
ici_parallelism = [
    args.ici_data_parallelism, args.ici_fsdp_parallelism,
    args.ici_tensor_parallelism
]

num_devices = pjrt.global_device_count()

assert np.product(dcn_parallelism) * np.product(
    ici_parallelism) == num_devices, f"Number of devices {num_devices} \
      does not match the product of the parallelism {np.product(dcn_parallelism) * np.product(ici_parallelism)}"

device_ids = np.arange(num_devices)

mesh_shape = (np.product(dcn_parallelism) * np.product(ici_parallelism), 1)
mesh = xs.Mesh(device_ids, mesh_shape, ('data', 'tensor'))

data_sharding = (0, 1)
parameter_sharding = (1, 0)

BATCH = num_devices * args.batch_size
D_EMB = args.embedding_dimension
D_FF = 4 * D_EMB
NUM_LAYERS = args.num_layers

parameters = 2 * D_FF * D_EMB * NUM_LAYERS
parameter_bytes = 2 * parameters
activation_bytes = 2 * (BATCH * (D_FF + D_EMB)) * NUM_LAYERS
memory_bytes = parameter_bytes + activation_bytes

print(
    f"total {memory_bytes/10**9} GB, parameters {parameter_bytes/10**9} GB, activations {activation_bytes/10**9} GB"
)


def gen_layer():
  return {
      "EMB2FF": 1e-4 * torch.randn(D_FF, D_EMB, dtype=torch.bfloat16),
      "FF2EMB": 1e-4 * torch.randn(D_FF, D_EMB, dtype=torch.bfloat16),
  }


def gen_layers():
  layers = []
  for _ in range(NUM_LAYERS):
    layer = gen_layer()

    def f(l):
      l = l.to(device)
      xs.mark_sharding(l, mesh, parameter_sharding)
      return l

    layer = {k: f(v) for k, v in layer.items()}
    layers.append(layer)
  return layers


def gen_data():
  return torch.rand(BATCH, D_EMB, dtype=torch.bfloat16, requires_grad=True)


def simple_timeit(f, tries=1, verbose=True):
  '''Simple utility to time a function for multiple runs'''

  s = datetime.datetime.now()
  f()
  e = datetime.datetime.now()
  outcomes = (e - s).total_seconds()
  return outcomes


def multiply_layer(in_act, in_layer):
  n1 = torch.nn.Sigmoid()
  M1 = n1(in_act @ in_layer["EMB2FF"].T)
  xs.mark_sharding(M1, mesh, data_sharding)
  n2 = torch.nn.Sigmoid()
  M2 = n2(M1 @ in_layer["FF2EMB"])
  xs.mark_sharding(M2, mesh, data_sharding)
  return M2


def multiply_layers(in_act, in_layers):
  x = in_act
  for i, layer in enumerate(in_layers):
    x = multiply_layer(x, layer)
    xs.mark_sharding(x, mesh, data_sharding)
  return x, in_layers


def multiply_layers_with_loss(in_act, in_layers):
  x, _ = multiply_layers(in_act, in_layers)
  return torch.sum(x)


def training_step(in_act, in_layers):
  multiply_layers_with_grad = multiply_layers_with_loss(in_act, in_layers)
  grad = torch.autograd.grad(multiply_layers_with_grad, in_act)
  return grad


torch.manual_seed(42)
tries = 5
device = xm.xla_device()
if args.profile:
  print("Profiler server started at port 9012")
  server = xp.start_server(9012)
data = gen_data().to(device)
xs.mark_sharding(data, mesh, data_sharding)
layers = gen_layers()
TFLOPs_per_device = parameters * 6 * BATCH / 10**12 / pjrt.global_device_count()
time = 0
# warm up step
with xp.StepTrace('train_sharding'):  # Warmup
  with xp.Trace('build_graph'):
    training_step(data, layers)
xm.mark_step()
for _ in range(tries):
  with xp.StepTrace('train_sharding'):
    with xp.Trace('build_graph'):
      time += simple_timeit(lambda: training_step(data, layers))
  xm.mark_step()
time /= tries
print(
    f"time is {time} seconds, TFLOP is {TFLOPs_per_device}, TFLOP/s is {TFLOPs_per_device/time}",
    flush=True)
