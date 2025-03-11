import jax
import jax.numpy as jnp
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import numpy as np
from typing import Tuple, Dict, Any
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
from functools import partial
import time


class RandomTensorDataset:
    def __init__(self, bs, num_tokens, input_dim, output_dim, element_count, seed=0, sharding=None):
        self.tensor_shape = (bs, num_tokens, input_dim)
        self.label_shape = (bs, num_tokens, output_dim)
        self.element_count = element_count
        self.sharding = sharding
        
        with jax.default_device(jax.devices('cpu')[0]):
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, element_count * 2)
            
            self.data = [jax.random.normal(keys[i], shape=self.tensor_shape, dtype=jnp.bfloat16) 
                         for i in range(0, element_count * 2, 2)]
            self.labels = [jax.random.randint(keys[i], shape=self.label_shape, minval=0, maxval=output_dim, dtype=jnp.int32) 
                           for i in range(1, element_count * 2, 2)]

    def __iter__(self):
        for tensor, label in zip(self.data, self.labels):
            if self.sharding:
                tensor = jax.device_put(tensor, self.sharding)
                label = jax.device_put(label, self.sharding)
            yield tensor, label


class FeedForwardNetwork(nn.Module):
    hidden_dim: int
    output_dim: int
    mesh: Mesh
    use_bias: bool = False
    
    def setup(self):
        self.dense1 = nn.Dense(
            features=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.bfloat16,
        )
        
        self.dense2 = nn.Dense(
            features=self.output_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.bfloat16,
        )
        
        self.dense3 = nn.Dense(
            features=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.bfloat16,
        )
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h1 = self.dense1(x)
        h3 = self.dense3(x)
        h = jax.nn.silu(h1 * h3)
        # h = jax.lax.with_sharding_constraint(h, NamedSharding(self.mesh, P('batch', None, 'model')))
        h = self.dense2(h)
        # h = jax.lax.with_sharding_constraint(h, NamedSharding(self.mesh, P('batch', None, None)))
        return h

class StackedFFN(nn.Module):
    num_layers: int
    hidden_dim: int
    output_dim: int
    mesh: Mesh
    use_bias: bool = False
    
    def setup(self):
        self.layers = [
            FeedForwardNetwork(
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                use_bias=self.use_bias,
                mesh=self.mesh,
            )
            for _ in range(self.num_layers)
        ]
        # self.out_proj = nn.Dense(
        #     features=self.out_channels,
        #     use_bias=self.use_bias,
        #     kernel_init=nn.initializers.normal(stddev=0.02),
        # )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        # x = self.out_proj(x)
        # x = jax.lax.with_sharding_constraint(x, NamedSharding(self.mesh, P('batch', None, None)))
        return x


def print_hlo(f, args, post_opt=False):
    if post_opt:
        print(f.lower(*args).compile().as_text())
    else:
        print(f.lower(*args).as_text("hlo"))

def main(num_layers=5, profile_path="profiles"):
    # Model configuration
    # model_axis = 4
    batch_size = 128
    seq_length = 512
    feature_dim = 8192
    hidden_dim = 28672
    # feature_dim = 512
    # hidden_dim = 1024
    num_layers = num_layers
    num_steps = 10

    # Mesh
    mesh = jax.make_mesh((jax.device_count(), 1), ('x', 'y'))
    
    # data_parallel_sharding = NamedSharding(mesh, P('batch', None, None))
    
    # Create model
    model = StackedFFN(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        output_dim=feature_dim,
        mesh=mesh
    )
    
    dataset = RandomTensorDataset(batch_size,
                                  seq_length,
                                  feature_dim,
                                  feature_dim,
                                  element_count=num_steps*2, # Init more data
                                  sharding=None)
    dataset_iter = iter(dataset)
    
    dummy_x, dummy_y = next(dataset_iter)
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = model.init(key, dummy_x)
    model.bind(params)

    # Shard inputs and weights
    for i in range(num_layers):
        layer_name = 'layers_' + str(i)
        print(params['params'][layer_name]['dense1']['kernel'].shape)
        print(params['params'][layer_name]['dense2']['kernel'].shape)
        print(params['params'][layer_name]['dense3']['kernel'].shape)
        k1 = params['params'][layer_name]['dense1']['kernel']
        params['params'][layer_name]['dense1']['kernel'] = jax.device_put(k1, NamedSharding(mesh, P(None, 'x')))
        k2 = params['params'][layer_name]['dense2']['kernel']
        params['params'][layer_name]['dense2']['kernel'] = jax.device_put(k2, NamedSharding(mesh, P('x', None)))
        k3 = params['params'][layer_name]['dense3']['kernel']
        params['params'][layer_name]['dense3']['kernel'] = jax.device_put(k3, NamedSharding(mesh, P(None, 'x')))
    
    @jax.jit
    def run_model(params, x):
        predictions = model.apply(params, x)
        return predictions
    
    print_hlo(run_model, (params, dummy_x), post_opt=True)
    dummy_inputs = []
    for i in range(num_steps):
        x, _ = next(dataset_iter)
        x = jax.device_put(x, NamedSharding(mesh, P(None, 'x', None)))
        dummy_inputs.append(x)
    # dummy_x = jax.device_put(dummy_x, NamedSharding(mesh, P('x', None, None)))
    with jax.profiler.trace(profile_path):
        for i in range(num_steps):
            start = time.time()
            # x, y = next(dataset_iter)
            # x, y = (dummy_x, dummy_y)
            output = run_model(params, dummy_inputs[i])
            # params, opt_state = train_step(params, x, y, opt_state)
            # print(params['params']['layers_0']['dense1']['kernel'].sharding)
            # print(grads['params']['layers_0']['dense1']['kernel'].sharding)
            jax.block_until_ready(output)
            print(f"Step {i}: step time {time.time() - start}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)