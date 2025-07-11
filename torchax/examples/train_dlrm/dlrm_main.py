import collections
from absl import app
import functools
import threading
import time
from typing import Any, List, Mapping
import torch
import torchax
from torchax.flax import FlaxNNModule
import torch.nn as nn
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
import jax
from absl import flags
from absl import logging
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.flax import embed
from dataloader import DummyCriteoDataset

FLAGS = flags.FLAGS

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8192, "Batch size.")
_NUM_DENSE_FEATURES = flags.DEFINE_integer(
    "num_dense_features", 13, "Number of dense features."
)
_EMBEDDING_SIZE = flags.DEFINE_integer("embedding_size", 8, "Embedding size.")


VOCAB_SIZES = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]
MULTI_HOT_SIZES = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]


# Define mesh
pd = P("x")
global_devices = jax.devices()
mesh = jax.sharding.Mesh(global_devices, "x")
global_sharding = jax.sharding.NamedSharding(mesh, pd)

class CrossNetwork(nn.Module):
    """
    Constructs the Cross Network of a DCN-V2 model.
    """
    def __init__(self, in_features, num_layers):
        super(CrossNetwork, self).__init__()
        self._num_layers = num_layers
        self._cross_layers = nn.ModuleList(
            nn.Linear(in_features, in_features) for _ in range(self._num_layers)
        )

    def forward(self, x_0):
        x_i = x_0
        for i in range(self._num_layers):
            # The core DCNv2 interaction
            x_i = x_0 * self._cross_layers[i](x_i) + x_i
        return x_i


class DLRM_DCNv2(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_embeddings_per_feature,
        dense_in_features,
        dense_arch_layer_sizes,
        over_arch_layer_sizes,
        dcn_num_layers,
        mesh=None,
        feature_specs = None,
        dummy_sparse_inputs = None,
    ):
        super(DLRM_DCNv2, self).__init__()

        # Sparse features embeddings
        flax_module = embed.SparseCoreEmbed(feature_specs=feature_specs, mesh=mesh, sharding_axis="data")
        env = torchax.default_env()
        module_layer = FlaxNNModule(env, flax_module, (store_lookups,), {})
        self.embedding_layer = module_layer

        # Dense features MLP (Bottom MLP)
        dense_layers = []
        for in_size, out_size in zip([dense_in_features] + dense_arch_layer_sizes[:-1], dense_arch_layer_sizes):
            dense_layers.append(nn.Linear(in_size, out_size))
            dense_layers.append(nn.ReLU())
        self.dense_mlp = nn.Sequential(*dense_layers)

        # DCNv2 Cross Network
        self.cross_network = CrossNetwork(
            in_features=embedding_dim + dense_arch_layer_sizes[-1],
            num_layers=dcn_num_layers,
        )

        # Top MLP
        top_layers = []
        num_sparse_features = len(num_embeddings_per_feature)
        # The input to the top MLP is the concatenation of the dense MLP output and the cross network output
        top_in_features = dense_arch_layer_sizes[-1] + (embedding_dim + dense_arch_layer_sizes[-1])

        for in_size, out_size in zip([top_in_features] + over_arch_layer_sizes[:-1], over_arch_layer_sizes):
            top_layers.append(nn.Linear(in_size, out_size))
            top_layers.append(nn.ReLU())
        self.top_mlp = nn.Sequential(*top_layers)

        # Final prediction layer
        self.final_linear = nn.Linear(over_arch_layer_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, dense_features, sparse_features):
        # Process dense features
        dense_x = self.dense_mlp(dense_features)

        # Process sparse features
        sparse_x = self.embedding_layer(sparse_features)
        sparse_x = torch.cat(sparse_x, dim=1)

        # Concatenate dense and sparse features for the cross network
        x_0 = torch.cat([dense_x, sparse_x], dim=1)

        # DCNv2 Interaction
        cross_out = self.cross_network(x_0)

        # Concatenate for Top MLP
        top_mlp_input = torch.cat([dense_x, cross_out], dim=1)

        # Top MLP
        top_out = self.top_mlp(top_mlp_input)

        # Final prediction
        logit = self.final_linear(top_out)
        prediction = self.sigmoid(logit)

        return prediction


def uniform_init(bound: float):
  def init(key, shape, dtype=jnp.float32):
    return jax.random.uniform(
        key,
        shape=shape,
        dtype=dtype,
        minval=-bound,
        maxval=bound
    )
  return init

def create_feature_specs(
    vocab_sizes: List[int],
) -> tuple[
    Mapping[str, embedding_spec.TableSpec],
    Mapping[str, embedding_spec.FeatureSpec],
]:
  """Creates the feature specs for the DLRM model."""
  table_specs = {}
  feature_specs = {}
  for i, vocab_size in enumerate(vocab_sizes):
    table_name = f"{i}"
    feature_name = f"{i}"
    bound = 0.5 #np.sqrt(1.0 / vocab_size)
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size,
        embedding_dim=_EMBEDDING_SIZE.value,
        initializer=uniform_init(bound),
        optimizer=embedding_spec.AdagradOptimizerSpec(learning_rate=0.01),
        combiner="sum",
        name=table_name,
        max_ids_per_partition=2048,
        max_unique_ids_per_partition=512,
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=(_BATCH_SIZE.value, 1),
        output_shape=(
            _BATCH_SIZE.value,
            _EMBEDDING_SIZE.value,
        ),
        name=feature_name,
    )
    feature_specs[feature_name] = feature_spec
    table_specs[table_name] = table_spec
  return table_specs, feature_specs




def main(argv):
    
    _, feature_specs = create_feature_specs(VOCAB_SIZES)
    embedding.prepare_feature_specs_for_training(
          feature_specs,
          global_device_count=jax.device_count(),
          num_sc_per_device=4,
      )
    embedding_dim = 32
    num_embeddings_per_feature = VOCAB_SIZES # Cardinality of each sparse feature
    dense_in_features = 13
    dense_arch_layer_sizes = [512, 256, 128]
    over_arch_layer_sizes = [1024, 1024, 512, 256]
    dcn_num_layers = 3
    batch_size = 8192
    dataset = DummyCriteoDataset(
            num_samples=_BATCH_SIZE.value,
            num_dense_features=13,
            vocab_sizes=VOCAB_SIZES,
            multi_hot_sizes=MULTI_HOT_SIZES,
        )


    dummy_data = next(iter(dataset))
    sparse_features = dummy_data['sparse_features']
    feature_weights = jax.tree_util.tree_map(
        lambda x: np.array(
            np.ones_like(x, shape=x.shape, dtype=np.float32)
        ),
        sparse_features
    )
    processed_sparse = embedding.preprocess_sparse_dense_matmul_input(
        sparse_features,
        feature_weights,
        feature_specs,
        mesh.local_mesh.size,
        mesh.size,
        num_sc_per_device=4,
        sharding_strategy="MOD",
        has_leading_dimension = False,
        allow_id_dropping=True,
        )[0]
    '''
    make_global_view = lambda x: jax.tree.map(
        lambda y: jax.make_array_from_process_local_data(
            self.global_sharding, y
        ),
        x,
    )
    processed_sparse = map(make_global_view, processed_sparse)
    '''
    dummy_sparse = processed_sparse

    # Create the model
    model = DLRM_DCNv2(
        embedding_dim=embedding_dim,
        num_embeddings_per_feature=num_embeddings_per_feature,
        dense_in_features=dense_in_features,
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        dcn_num_layers=dcn_num_layers,
        mesh=mesh,
        feature_specs=feature_specs,
        dummy_sparse_inputs=dummy_sparse
    )

if __name__ == "__main__":
    app.run(main)
