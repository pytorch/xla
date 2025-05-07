import itertools
import random
import unittest
import sys
from itertools import product

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.sparse_adam import SparseAdam
from torch.testing._internal.common_utils import TestCase

from absl.testing import parameterized
from torch_xla.experimental.sparse import Embedding, EmbeddingBag, embedding as xla_embedding, embedding_bag as xla_embedding_bag, SparseCOOTensor

index_dtypes = {torch.int32, torch.int64}
float_dtypes = {torch.float32, torch.float16, torch.bfloat16}

xla_device = torch_xla.device()

# def _ref_embedding(input, init_args*, **init_kwargs):
# upstream_embed = nn.Embedding(*init_args, **init_kwargs)
# return upstream_embed(input)


class TestEmbedding(TestCase):

  def test_embedding_sparse_basic(self):
    embedding = Embedding(10, 20, sparse=True, device=xla_device)
    input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]],
                         dtype=torch.long,
                         device=xla_device)
    output = embedding(input)
    output.max().backward()
    self.assertTrue(embedding.weight.grad.is_sparse)
    self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

  def test_embedding_functional(self):
    torch_input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long)
    torch_weights = torch.randn(10, 20, requires_grad=True)
    xla_input = torch_input.to(xla_device)
    xla_weights = torch_weights.detach().to(xla_device).requires_grad_(True)
    torch_out = F.embedding(torch_input, torch_weights, sparse=True)
    xla_out = xla_embedding(xla_input, xla_weights, sparse=True)
    self.assertEqual(xla_out, torch_out)
    torch_out.max().backward()
    xla_out.max().backward()
    self.assertTrue(xla_weights.grad.is_sparse)
    self.assertEqual(xla_weights.grad, torch_weights.grad)

  def test_embedding_optimizer(self):
    context_size = 2
    embedding_dim = 10
    # We will use Shakespeare Sonnet 2, first few lines
    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:""".split()
    ngrams = [([test_sentence[i - j - 1]
                for j in range(context_size)], test_sentence[i])
              for i in range(context_size, len(test_sentence))]
    vocab = set(test_sentence)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    model = Embedding(vocab_size, context_size, sparse=True).to(xla_device)

    optimizer = SparseAdam(model.parameters(), lr=0.001)
    for _ in range(10):
      total_loss = 0
      for context, _ in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context],
                                    dtype=torch.long).to(xla_device)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = log_probs.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        xm.mark_step()
    # TODO: inspect optimizer context assert update has happend.


# repeat for Bag

if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
