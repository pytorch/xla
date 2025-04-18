import itertools
import random
import unittest
import sys
from itertools import product

import torch
import torch_xla
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

from absl.testing import parameterized
from torch_xla.experimental.sparse import Embedding, EmbeddingBag, embedding as xla_embedding, embedding_bag as xla_embedding_bag, SparseCOOTensor

index_dtypes = {torch.int32, torch.int64}
float_dtypes = {torch.float32, torch.float16, torch.bfloat16}

xla_device = torch_xla.device()

# def _ref_embedding(input, init_args*, **init_kwargs):
# upstream_embed = nn.Embedding(*init_args, **init_kwargs)
# return upstream_embed(input)


class TestEmbeddingNN(TestCase):

  def test_embedding_sparse_basic(self):
    embedding = Embedding(10, 20, sparse=True, device=xla_device)
    input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]],
                         dtype=torch.long,
                         device=xla_device)
    output = embedding(input)
    output.max().backward()
    self.assertTrue(embedding.weight.grad.is_sparse)
    self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)


# test functional variant
# test w/ optimizer
# repeat for Bag

if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
