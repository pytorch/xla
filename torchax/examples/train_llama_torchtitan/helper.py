import time
import jax
from jax.tree_util import tree_map
from jax.sharding import NamedSharding
from torchax import interop

P = jax.sharding.PartitionSpec


def compile_step_func(step, weights, buffers, opt_state, args, label, mesh):
  step, weights, buffers, opt_state, args, label = interop.jax_view(
      (step, weights, buffers, opt_state, args, label))
  wshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, weights)
  bshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, buffers)
  oshardings = tree_map(
      lambda a: a.sharding if isinstance(a, jax.Array) else None, opt_state)
  print('Start compiling')
  start = time.perf_counter()
  lowered = jax.jit(
      step,
      donate_argnums=(0, 2),
      #in_shardings=shardings,
      out_shardings=(NamedSharding(mesh, P()), wshardings, oshardings),
  ).lower(weights, buffers, opt_state, args, label)
  #print(lowered.as_text())
  # import pdb; pdb.set_trace()
  print('program size:', len(lowered.as_text()) / 1e6, 'm chars')
  step_compiled = lowered.compile()
  end = time.perf_counter()
  print('End compiling', end - start)
  compile_time = end - start
  for co in step_compiled.cost_analysis():
    print('Flops', co['flops'])
    print('GB accessed', co['bytes accessed'] / 1e9)
  return interop.torch_view(step_compiled)
