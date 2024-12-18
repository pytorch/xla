import os
import time
import logging
from typing import Tuple
from collections import defaultdict
import functools


def _setup_default_env():
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
import torch_xla2.train
from torch_xla2.interop import jax_view, torch_view, JittableModule
import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental import mesh_utils
import optax

from torchtitan.models.llama import llama3_configs
from torchtitan.models.llama import model as titan

P = jax.sharding.PartitionSpec

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


class Trainer:

    def __init__(self, mesh):
        self.mesh = mesh
        self.x_sharding = jax.sharding.NamedSharding(self.mesh, P(global_axis))
        self.replicated = jax.sharding.NamedSharding(self.mesh, P())

    def fit(self, model, loss_fn, data_loader):
        xla_env = torch_xla2.default_env()
        jax.config.update('jax_enable_x64', False)
        xla_env._mesh = self.mesh
        xla_env.use_flash_attention = True

        
        model.to('jax')
        jittable_mod = JittableModule(model)

        # split the params to the n devices

        def model_fn(weights, buffers, args):
            return jittable_mod.functional_call('forward', weights, buffers, args)

        jax_optimizer = optax.adamw(0.001)
        opt_state = torch_xla2.interop.call_jax(jax_optimizer.init, jittable_mod.params)

        train_step = torch_xla2.train.make_train_step(
            model_fn, loss_fn, jax_optimizer,
            remat_policy=jax.checkpoint_policies.nothing_saveable,
            mark_fsdp_sharding_axis='fsdp')

        print('Begining training')
        s = time.perf_counter()
        jax.profiler.start_trace('/tmp/tensorboard')
        print('start training')
        min_loop_time = 10000
        for i, item in enumerate(data_loader):
            inputs, labels = item
            # Move them to jax device
            inputs = inputs.to('jax')
            labels = labels.to('jax')

            # Shard them on batch dim for fsdp
            inputs.apply_(sharded_device_put, self.x_sharding)
            labels.apply_(sharded_device_put, self.x_sharding)

            print('INPUT shape', inputs.shape)
            step_start = time.perf_counter()
            loss, jittable_mod.params, opt_state = train_step(
                jittable_mod.params, jittable_mod.buffers, opt_state, inputs, labels)
            # wait for iteration to finish to measure time 
            jax.block_until_ready((loss, jittable_mod.params))
            step_end = time.perf_counter()
            print(i, 'loss', loss, 'step latency: ', step_end - step_start)
            loop_time = step_end - step_start
            min_loop_time = min(min_loop_time, loop_time)
            print('======')
            if i >= 3:
                break
        jax.profiler.stop_trace()
        return min_loop_time


def create_sharded_weights(state_dict, sharding):
    res = {}
    env = torch_xla2.default_env()
    for name, weight_meta in state_dict.items():
        with jax.default_device(jax.devices('cpu')[0]):
            weight_torch = torch.randn(
              weight_meta.shape,
              dtype=weight_meta.dtype)
            # weight_jax is jax array
            weight_jax = env.to_xla(weight_torch).jax()
        res[name] = env.j2t_iso(jax.make_array_from_callback(
          weight_jax.shape, sharding, lambda a: weight_jax[a]
        ))
    return res


def fake_dataloader(size, seqlen, batch_size):
  for _ in range(size):
    x = torch.randint(0, 32000, (batch_size, seqlen), device='cpu')
    yield x, (x + 1) % 32000


def main(
    model_type='8B',
    batch_size=8,
    seqlen=2048,
    override_num_layers=-1,
):
    torch_xla2.enable_globally()
    #logging.getLogger("jax").setLevel(logging.DEBUG)
    print(f"Running with parameters {locals()}")

    mesh = jax.make_mesh((len(jax.local_devices()), ), ('fsdp', ))
    sharding = jax.sharding.NamedSharding(mesh, P('fsdp')) 

    env = torch_xla2.default_env()
    env.config.use_tpu_flash_attention = True
    env.config.shmap_flash_attention = True
    env._mesh = mesh  # this is the mesh used by flash attention pallas kernel

    args = llama3_configs[model_type]
    # Note: torchtitan's upstream config did not specify this value
    args.vocab_size = 128256
    if override_num_layers > 0:
        args.n_layers = override_num_layers

    # Note: because a single device don't have enough HBM memory
    # nor enough CPU memory to hold the parameters. We instantiate
    # the model on meta then manually initialize then shard each param
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('meta'):
        gpt = titan.Transformer(args)
        gpt.to(torch.bfloat16)

    state_dict = create_sharded_weights(gpt.state_dict(), sharding)
    gpt.load_state_dict(state_dict, assign=True)
    
    train_loader = fake_dataloader(10, seqlen, batch_size)

    def loss_fn(logits, y):
        num_tokens = logits.shape[-1]
        logits = logits.reshape(-1, num_tokens)
        y = y.reshape(-1)
        return torch.nn.functional.cross_entropy(
            logits, y)

    with mesh:
        trainer = Trainer()
        return trainer.fit(
            gpt, 
            loss_fn,
            train_loader 
        )


if __name__ == '__main__':
    import fire
    fire.Fire(main)
