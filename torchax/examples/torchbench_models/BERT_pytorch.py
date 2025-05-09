import torch
import time
import torchax
import torchax.interop
import os
import importlib
import sys
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# NOTE: replace this patch below with your installation
TORCH_BENCH_PATH = os.path.expanduser('~/git/qihqi/benchmark')
# If your directory looks like this_file.py, benchmark/
sys.path.append(TORCH_BENCH_PATH)
model_name = "torchbenchmark.models.BERT_pytorch"  # replace this by the name of the model you're working on
module = importlib.import_module(model_name)
benchmark_cls = getattr(module, "Model", None)
benchmark = benchmark_cls(
    test="eval", device="cpu")  # test = train or eval device = cuda or cpu

model, example = benchmark.get_module()

env = torchax.default_env()
env.config.debug_print_each_op = False
model = env.to_xla(model)
example = env.to_xla(example)
with env:
  start = time.perf_counter()
  print(model(*example))
  end = time.perf_counter()
  print('Eager mode time', end - start)


def func_call(state, example):
  return torch.func.functional_call(model, state, example, tie_weights=False)


jitted = torchax.interop.jax_jit(func_call)
start = time.perf_counter()
print(func_call(model.state_dict(), example))
end = time.perf_counter()
print('Jitted mode time', end - start)
