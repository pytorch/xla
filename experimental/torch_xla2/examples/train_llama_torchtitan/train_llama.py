import os
import time
import logging
from typing import Tuple
from collections import defaultdict
import functools

def _setup_default_env():
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
  os.environ.setdefault('ALLOW_MULTIPLE_LIBTPU_LOAD', '1')
  # only need for tpu v4
  # os.environ.setdefault('TPU_MEGACORE', 'megacore_dense')
  tpu_args = "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
  os.environ.setdefault('LIBTPU_INIT_ARGS', tpu_args)

_setup_default_env()

import torch
import torch.nn.functional
from torch.utils import _pytree as pytree

import torch_xla2
import torch_xla2.interop
from torch_xla2.interop import jax_view, torch_view, JittableModule
import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental import mesh_utils
import optax

from torchtitan.models.llama import llama3_configs
from torchtitan.models.llama import model as titan

P = jax.sharding.PartitionSpec



SEQLEN = 8192
BATCH = 8
global_axis: Tuple[str, str] = ('fsdp', )
num_global_devices = jax.device_count()
num_local_devices = jax.local_device_count()
num_partitions = (num_global_devices, )


def sharded_device_put(tensor, sharding):
    if isinstance(tensor, tuple):
        return tuple(sharded_device_put(t, sharding) for t in tensor)

    if num_global_devices == num_local_devices:
        return jax.device_put(tensor, sharding)

    shape = tensor.shape
    x_split = [jax.device_put(tensor[i], device) for device, i in sharding.addressable_devices_indices_map(shape).items()]
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
        return torch_xla2.interop.call_jax(
            jax.lax.with_sharding_constraint,
            x,
            self.sharding,
        )

def print_shapes(pyt):
    for p in pytree.tree_flatten(pyt)[0]:
        if hasattr(p, 'shape'):
            print(p.shape, p.dtype)


class Module(torch.nn.Module):

    def __init__(self, inner):
        super().__init__()
        self.inner = FSDPv2(inner)

    def training_step(self, data, batch_id):
        x, y = data
        logits = self.inner(x)
        num_tokens = logits.shape[-1]
        logits = logits.reshape(-1, num_tokens)
        y = y.reshape(-1)
        return torch.nn.functional.cross_entropy(
            logits, y)
    

class Trainer:

    def __init__(self):
        self.mesh = jax.sharding.Mesh(
            mesh_utils.create_device_mesh(num_partitions),
            axis_names=global_axis,
        )
        self.x_sharding = jax.sharding.NamedSharding(self.mesh, P(global_axis))
        self.replicated = jax.sharding.NamedSharding(self.mesh, P())

    def _shard_fsdp_style(self, state_dict, sharding=None):
        if sharding is None:
            sharding = self.x_sharding
        def move_one_tensor(x):
            jval = torch_xla2.tensor.t2j(x)
            return sharded_device_put(jval, sharding)

        if isinstance(state_dict, torch.Tensor):
            return move_one_tensor(state_dict)
        res = {}
        for k, v in sorted(state_dict.items()):
            res[k] = move_one_tensor(v)
        return res

    def fit(self, lightning_mod, data_loader):
        xla_env = torch_xla2.default_env()
        jax.config.update('jax_enable_x64', False)
        xla_env._mesh = self.mesh
        xla_env.use_flash_attention = True

        jittable_mod = JittableModule(lightning_mod)
        jax_params = self._shard_fsdp_style(jittable_mod.params) 
        jax_buffers = self._shard_fsdp_style(jittable_mod.buffers)

        @jax.checkpoint
        def lightning_mod_loss(
            weights: jax.Array, buffers: jax.Array, data: jax.Array, batch_id):
            """returns loss"""
            with jax.named_scope("Computing_loss"):
                weights, buffers, data = torch_view((weights, buffers, data))
                # NOTE: these is needed because the original model
                # did not register those as persistent buffer
                with xla_env:
                    loss = jittable_mod.functional_call(
                        'training_step',
                        weights, buffers, data, batch_id)
                return jax_view(loss)

        jax_optimizer = optax.adamw(0.001)

        opt_state = jax_optimizer.init(jax_params)
        grad_fn = jax.value_and_grad(lightning_mod_loss)

        opt_state_sharding = jax.tree_util.tree_map(lambda p : p.sharding, opt_state)

        print('Begining training')

        @functools.partial(
            jax.jit, 
            donate_argnums=(0, 2),
        )
        def step(jax_weights, jax_buffers, optimizer_state, xla_data, bid):
            print('Tracing inside of step')
            with jax.named_scope("Computing_loss_and_grad"):
                loss, grads = grad_fn(jax_weights, jax_buffers, xla_data, bid)
            with jax.named_scope("optimizer_updates"):
                updates, opt_state = jax_optimizer.update(
                    grads, optimizer_state, jax_weights)
                jax_weights = optax.apply_updates(jax_weights, updates)
            return loss, jax_weights, opt_state

        total_param_size = 0
        for k, v in jax_params.items():
            total_param_size += v.size

        print('Total number of params: ', total_param_size)

        print('Start compiling')
        start = time.perf_counter()
        lowered = step.lower(
            jax_params, jax_buffers, opt_state, 
            (jax.ShapeDtypeStruct((BATCH, SEQLEN), jnp.dtype('int32'), sharding=self.x_sharding), 
             jax.ShapeDtypeStruct((BATCH, SEQLEN), jnp.dtype('int32'), sharding=self.x_sharding)),
            0
        )
        # print(lowered.as_text())
        print('program size:', len(lowered.as_text()) / 1e6, 'm chars')
        step_compiled  = lowered.compile()
        end = time.perf_counter()
        compile_time = end - start
        print('End compiling', compile_time)

        for co in step_compiled.cost_analysis():
            print('flops counter:', co['flops'])

        s = time.perf_counter()
        jax.profiler.start_trace('/tmp/tensorboard')
        print('start training')
        min_loop_time = 10000
        for i, item in enumerate(data_loader):
            inputs, labels = sharded_device_put(jax_view(xla_env.to_xla(item)), 
                                            self.x_sharding)
            print('INPUT shape', inputs.shape)

            step_start = time.perf_counter()
            loss, jax_params, opt_state = step_compiled(
                jax_params, jax_buffers, opt_state, (inputs, labels), 0)
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



def fake_dataloader(size, seqlen, batch_size):
  for _ in range(size):
    x = torch.randint(0, 32000, (batch_size, seqlen), device='cpu')
    yield x, (x + 1) % 32000


def main(
    model_type='8B',
    batch_size=8,
    seqlen=2048,
    mode='regular', 
):
    logging.getLogger("jax").setLevel(logging.DEBUG)
    print(f"Running with parameters {locals()}")
    global SEQLEN
    global BATCH
    SEQLEN = seqlen
    BATCH = batch_size

    mesh = jax.make_mesh((len(jax.local_devices()), ), ('fsdp', ))
    env = torch_xla2.default_env()
    env.config.use_tpu_flash_attention = use_flash_attention
    env.config.shmap_flash_attention = use_flash_attention

    args = llama3_configs[model_type]
    #with torch.device('meta'):
    gpt = titan.Transformer(args)

    light_mod = Module(gpt)
    light_mod.to(torch.bfloat16)

    train_loader = fake_dataloader(10, seqlen, batch_size)

    with mesh:
        trainer = Trainer()
        return trainer.fit(
            light_mod, 
            train_loader 
        )


if __name__ == '__main__':
    import fire
    fire.Fire(main)
