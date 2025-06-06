from typing import Tuple
import time
import torchax
from torchax.interop import jax_view, torch_view, JittableModule
import jax
import optax
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
import functools
from torch.utils import _pytree as pytree

Mesh = jax.sharding.Mesh
P = jax.sharding.PartitionSpec

SEQLEN = 8192
BATCH = 8
global_axis: Tuple[str, str] = ('fsdp',)
num_global_devices = jax.device_count()
num_local_devices = jax.local_device_count()
num_partitions = (num_global_devices,)
#SEQLEN = 512

import torch


def group_data(dataloader, block_size):
  """yields tuple of inputs, label with seqlen == block_size"""

  tally = 0
  inputs = []
  labels = []

  for line in dataloader:
    x, y = line['input_ids'], line['labels']
    inputs.append(x)
    labels.append(y)
    batch, seqlen = x.shape
    tally += seqlen
    if tally > block_size:
      inputs_stacked = torch.concat(inputs, dim=-1)
      inputs_stacked.resize_((batch, block_size))
      labels_stacked = torch.concat(labels, dim=-1)
      labels_stacked.resize_((batch, block_size))
      yield inputs_stacked, labels_stacked
      tally = 0
      inputs = []
      labels = []


def sharded_device_put(tensor, sharding):
  if isinstance(tensor, tuple):
    return tuple(sharded_device_put(t, sharding) for t in tensor)

  if num_global_devices == num_local_devices:
    return jax.device_put(tensor, sharding)

  shape = tensor.shape
  x_split = [
      jax.device_put(tensor[i], device)
      for device, i in sharding.addressable_devices_indices_map(shape).items()
  ]
  return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


class FSDPv2(torch.nn.Module):

  def __init__(self, mod):
    super().__init__()
    self.mod = mod
    self.mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh(num_partitions),
        axis_names=global_axis,
    )
    self.sharding = jax.sharding.NamedSharding(self.mesh, P(*global_axis))

  def forward(self, *args):
    args = list(args)
    args[0] = self.shard(args[0])
    res = self.mod(*args)
    return self.shard(res)

  def shard(self, x):
    return torchax.interop.call_jax(
        jax.lax.with_sharding_constraint,
        x,
        self.sharding,
    )


def print_shapes(pyt):
  for p in pytree.tree_flatten(pyt)[0]:
    if hasattr(p, 'shape'):
      print(p.shape, p.dtype)


class JaxTrainer:

  def __init__(self, use_fori):
    self.use_fori = use_fori
    self.mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh(num_partitions),
        axis_names=global_axis,
    )
    self.x_sharding = jax.sharding.NamedSharding(self.mesh, P(global_axis))
    self.y_sharding = jax.sharding.NamedSharding(self.mesh, P(*global_axis))
    self.replicated = jax.sharding.NamedSharding(self.mesh, P())

  def torch_opt_to_jax_opt(self, torch_opt):
    # TODO: Can convert optimizer instead of using a jax one
    return optax.adamw(0.01)

  def fit_model_fori(self, gpt_mod, data_loader):
    xla_env = torchax.default_env()
    jax.config.update('jax_enable_x64', False)
    xla_env._mesh = self.mesh
    xla_env.use_flash_attention = True

    weights = gpt_mod.weights

    jax_params = {}
    for k, v in weights.items():
      sharding = self.y_sharding if k == 'block' else self.x_sharding
      print(k, sharding)
      jax_params[k] = self._shard_fsdp_style(v, sharding)

    print('ALL weights ===')
    for x in jax.tree_util.tree_flatten(jax_params)[0]:
      print(x.shape, x.sharding)
    print(' ===')

    @jax.checkpoint
    def loss(jax_params, data):
      data = jax.lax.with_sharding_constraint(data, self.x_sharding)  # fsdpv2
      x, y = data
      res = torchax.interop.call_torch(gpt_mod.forward_with_weights, jax_params,
                                       x)
      res = jax.lax.with_sharding_constraint(res, self.x_sharding)
      return jnp.mean(
          optax.losses.softmax_cross_entropy_with_integer_labels(res, y))

    grad_fn = jax.value_and_grad(loss)
    jax_optimizer = optax.adamw(0.01)
    opt_state = jax_optimizer.init(jax_params)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(jax_weights, opt_state, data):
      with jax.named_scope('compute_gradient'):
        loss, gradient = grad_fn(jax_weights, data)
      with jax.named_scope("optimizer_updates"):
        updates, opt_state = jax_optimizer.update(gradient, opt_state,
                                                  jax_weights)
        jax_weights = optax.apply_updates(jax_weights, updates)
      return loss, jax_weights, opt_state

    print('Start compiling')
    start = time.perf_counter()
    lowered = step.lower(
        jax_params,
        opt_state,
        (jax.ShapeDtypeStruct(
            (BATCH, SEQLEN), jnp.dtype('int32'), sharding=self.x_sharding),
         jax.ShapeDtypeStruct(
             (BATCH, SEQLEN), jnp.dtype('int32'), sharding=self.x_sharding)),
    )
    # print(lowered.as_text())
    print('program size:', len(lowered.as_text()) / 1e6, 'm chars')
    step_compiled = lowered.compile()
    end = time.perf_counter()
    print('End compiling', end - start)
    compile_time = end - start

    for co in step_compiled.cost_analysis():
      print('flops counter:', co['flops'])

    s = time.perf_counter()
    jax.profiler.start_trace('/tmp/tensorboard')
    print('start training')
    min_loop_time = 10000
    for i, item in enumerate(group_data(data_loader, SEQLEN)):
      inputs, labels = sharded_device_put(
          jax_view(xla_env.to_xla(item)), self.x_sharding)
      print('INPUT shape', inputs.shape)

      step_start = time.perf_counter()
      loss, jax_params, opt_state = step_compiled(jax_params, opt_state,
                                                  (inputs, labels))
      jax.block_until_ready((loss, jax_params))
      step_end = time.perf_counter()
      print(i, 'loss', loss, 'step latency: ', step_end - step_start)
      min_loop_time = min(min_loop_time, step_end - step_start)
      print('======')
      if i >= 3:
        break
    jax.profiler.stop_trace()
    return min_loop_time, compile_time

  def _shard_fsdp_style(self, state_dict, sharding=None):
    if sharding is None:
      sharding = self.x_sharding

    def move_one_tensor(x):
      env = torchax.default_env()
      jval = env.t2j_copy(x)
      return sharded_device_put(jval, sharding)

    if isinstance(state_dict, torch.Tensor):
      return move_one_tensor(state_dict)
    res = {}
    for k, v in sorted(state_dict.items()):
      res[k] = move_one_tensor(v)
    return res

  def fit(self, lightning_mod, data_loader):

    xla_env = torchax.default_env()
    jax.config.update('jax_enable_x64', False)
    xla_env._mesh = self.mesh
    xla_env.use_flash_attention = True

    jittable_mod = JittableModule(lightning_mod)
    jax_params = self._shard_fsdp_style(jittable_mod.params)
    jax_buffers = self._shard_fsdp_style(jittable_mod.buffers)

    @jax.checkpoint
    def lightning_mod_loss(weights: jax.Array, buffers: jax.Array,
                           data: jax.Array, batch_id):
      """returns loss"""
      with jax.named_scope("Computing_loss"):
        weights, buffers, data = torch_view((weights, buffers, data))
        # NOTE: these is needed because the original model
        # did not register those as persistent buffer
        with xla_env:
          loss = jittable_mod.functional_call('training_step', weights, buffers,
                                              data, batch_id)
        return jax_view(loss)

    jax_optimizer = self.torch_opt_to_jax_opt(
        lightning_mod.configure_optimizers())

    opt_state = jax_optimizer.init(jax_params)
    grad_fn = jax.value_and_grad(lightning_mod_loss)

    opt_state_sharding = jax.tree_util.tree_map(lambda p: p.sharding, opt_state)

    print('Begining training')

    # NOTE: explicitly set sharding so the sharding of opt_state wont change
    # if it changes, it would trigger recompile
    @functools.partial(
        jax.jit,
        donate_argnums=(0, 2),
        #in_shardings=(self.x_sharding, self.x_sharding, opt_state_sharding, self.x_sharding, self.replicated),
        #out_shardings=(self.replicated, self.x_sharding, opt_state_sharding),
    )
    def step(jax_weights, jax_buffers, optimizer_state, xla_data, bid):
      print('Tracing inside of step')
      with jax.named_scope("Computing_loss_and_grad"):
        loss, grads = grad_fn(jax_weights, jax_buffers, xla_data, bid)
      with jax.named_scope("optimizer_updates"):
        updates, opt_state = jax_optimizer.update(grads, optimizer_state,
                                                  jax_weights)
        jax_weights = optax.apply_updates(jax_weights, updates)
      return loss, jax_weights, opt_state

    total_param_size = 0
    for k, v in jax_params.items():
      total_param_size += v.size

    print('Total number of params: ', total_param_size)
    # print(jax.jit(jax.grad(lightning_mod_loss)).lower(
    #     jax_params, jax_buffers,
    #     (jax.ShapeDtypeStruct((8, SEQLEN), jnp.dtype('int32')),
    #      jax.ShapeDtypeStruct((8, SEQLEN), jnp.dtype('int32'))),
    #     0
    # ).as_text())

    print('Start compiling')
    start = time.perf_counter()
    lowered = step.lower(
        jax_params, jax_buffers, opt_state,
        (jax.ShapeDtypeStruct(
            (8, SEQLEN), jnp.dtype('int32'), sharding=self.x_sharding),
         jax.ShapeDtypeStruct(
             (8, SEQLEN), jnp.dtype('int32'), sharding=self.x_sharding)), 0)
    # print(lowered.as_text())
    print('program size:', len(lowered.as_text()) / 1e6, 'm chars')
    step_compiled = lowered.compile()
    end = time.perf_counter()
    compile_time = end - start
    print('End compiling', compile_time)

    for co in step_compiled.cost_analysis():
      print('flops counter:', co['flops'])

    s = time.perf_counter()
    jax.profiler.start_trace('/tmp/tensorboard')
    print('start training')
    min_loop_time = 10000
    for i, item in enumerate(group_data(data_loader, SEQLEN)):
      inputs, labels = sharded_device_put(
          jax_view(xla_env.to_xla(item)), self.x_sharding)
      print('INPUT shape', inputs.shape)

      step_start = time.perf_counter()
      loss, jax_params, opt_state = step_compiled(jax_params, jax_buffers,
                                                  opt_state, (inputs, labels),
                                                  0)
      jax.block_until_ready((loss, jax_params))
      step_end = time.perf_counter()
      print(i, 'loss', loss, 'step latency: ', step_end - step_start)
      loop_time = step_end - step_start
      min_loop_time = min(min_loop_time, loop_time)
      print('======')
      if i >= 2:
        break
    jax.profiler.stop_trace()
    return min_loop_time, compile_time
