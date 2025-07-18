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
    context_size = 5
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
    cloned_weight = model.weight.clone().detach()

    optimizer = SparseAdam(model.parameters(), lr=0.9)
    optimizer.state
    total_loss = 0
    for _ in range(5):
      for context, _ in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context],
                                    dtype=torch.long).to(xla_device)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = log_probs.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        xm.mark_step()
    self.assertNotEqual(total_loss, 0)
    self.assertNotEqual(cloned_weight, model.weight)


class TestEmbeddingBag(TestCase):

  def test_embedding_bag_sparse_basic(self):
    embedding = EmbeddingBag(10, 20, sparse=True, device=xla_device)
    input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]],
                         dtype=torch.long,
                         device=xla_device)
    output = embedding(input)
    output.max().backward()
    self.assertTrue(embedding.weight.grad.is_sparse)
    self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)

  def test_embedding_bag_functional(self):
    torch_input = torch.tensor([[0, 2, 4, 5], [4, 3, 0, 9]], dtype=torch.long)
    torch_weights = torch.randn(10, 20, requires_grad=True)
    xla_input = torch_input.to(xla_device)
    xla_weights = torch_weights.detach().to(xla_device).requires_grad_(True)
    torch_out = F.embedding_bag(torch_input, torch_weights, sparse=True)
    xla_out = xla_embedding_bag(xla_input, xla_weights, sparse=True)
    self.assertEqual(xla_out, torch_out)
    torch_out.max().backward()
    xla_out.max().backward()
    self.assertTrue(xla_weights.grad.is_sparse)
    self.assertEqual(xla_weights.grad, torch_weights.grad)

  def test_embedding_bag_optimizer(self):
    context_size = 10
    embedding_dim = 5
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

    model = EmbeddingBag(vocab_size, embedding_dim, sparse=True).to(xla_device)
    cloned_weight = model.weight.clone().detach()

    optimizer = SparseAdam(model.parameters(), lr=0.9)
    total_loss = 0
    offsets = torch.tensor([0, 5], dtype=torch.int64, device=xla_device)
    for _ in range(5):
      for context, _ in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context],
                                    dtype=torch.long).to(xla_device)
        model.zero_grad()
        log_probs = model(context_idxs, offsets)
        loss = log_probs.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        xm.mark_step()
    self.assertNotEqual(total_loss, 0)
    self.assertNotEqual(cloned_weight, model.weight)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
