import itertools
import random
import unittest
from itertools import product

import torch
import torch_xla
import torch.nn as nn
import torch.nn.functional as FA
from absl.testing import parameterized
from torch_xla.experimental import sparse as sparse_xla

index_dtypes = {torch.int32, torch.int64}
float_dtypes = {torch.float32, torch.float16, torch.bfloat16}

device = torch_xla.device()


def functional_embedding(impl):

  def f(input):
    impl()

  @parameterized.parameters(*index_dtypes)
  @parameterized.parameters(*float_dtypes)
  def test_embedding_sparse_basic(self, index_dtype, float_dtype):
    torch_embedding = nn.Embedding(10, 20, sparse=True, dtype=float_dtype)
    xla_embedding = sparse_xla.Embedding.from_pretrained(
        torch_embedding.weight.clone().detach().to(device), sparse=True)
    input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=index_dtype)
    input_xla = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]],
                             dtype=index_dtype,
                             device=device)
    ref_out = torch_embedding(input)
    xla_out = xla_embedding(input_xla)
    self.assertTrue(torch.allclose(ref_out, xla_out))
    ref_out.sum().backward()
    xla_out.sum().backward()
    torch_weight_grad = torch_embedding.weight.grad
    xla_weight_grad = xla_embedding.weight.grad
