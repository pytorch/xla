import unittest
import torch
import torch.nn as nn
import torchax
from torchax.checkpoint import _to_torch, _to_jax
import optax
import tempfile
import os
import jax
import jax.numpy as jnp
import shutil


class CheckpointTest(unittest.TestCase):

  def test_save_and_load_jax_style_checkpoint(self):
    model = torch.nn.Linear(10, 20)
    optimizer = optax.adam(1e-3)

    torchax.enable_globally()
    params_jax, _ = torchax.extract_jax(model)
    opt_state = optimizer.init(params_jax)
    torchax.disable_globally()

    epoch = 1
    state = {
        'model': model.state_dict(),
        'opt_state': opt_state,
        'epoch': epoch,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'checkpoint')
      torchax.save_checkpoint(state, path, step=epoch)
      loaded_state_jax = torchax.load_checkpoint(path)
      loaded_state = _to_torch(loaded_state_jax)

      self.assertEqual(state['epoch'], loaded_state['epoch'])

      # Compare model state_dict
      for key in state['model']:
        self.assertTrue(
            torch.allclose(state['model'][key], loaded_state['model'][key]))

      # Compare optimizer state
      original_leaves = jax.tree_util.tree_leaves(state['opt_state'])
      loaded_leaves = jax.tree_util.tree_leaves(loaded_state['opt_state'])
      for original_leaf, loaded_leaf in zip(original_leaves, loaded_leaves):
        if isinstance(original_leaf, (jnp.ndarray, jax.Array)):
          # Convert loaded leaf to numpy array for comparison if it is a DeviceArray
          self.assertTrue(jnp.allclose(original_leaf, jnp.asarray(loaded_leaf)))
        else:
          self.assertEqual(original_leaf, loaded_leaf)

  def test_load_pytorch_style_checkpoint(self):
    model = torch.nn.Linear(10, 20)
    optimizer = optax.adam(1e-3)

    torchax.enable_globally()
    params_jax, _ = torchax.extract_jax(model)
    opt_state = optimizer.init(params_jax)
    torchax.disable_globally()

    epoch = 1
    state = {
        'model': model.state_dict(),
        'opt_state': opt_state,
        'epoch': epoch,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'checkpoint.pt')
      torch.save(state, path)
      loaded_state_jax = torchax.load_checkpoint(path)

      # convert original state to jax for comparison
      state_jax = _to_jax(state)

      self.assertEqual(state_jax['epoch'], loaded_state_jax['epoch'])

      # Compare model state_dict
      for key in state_jax['model']:
        self.assertTrue(
            jnp.allclose(state_jax['model'][key],
                         loaded_state_jax['model'][key]))

      # Compare optimizer state
      original_leaves = jax.tree_util.tree_leaves(state_jax['opt_state'])
      loaded_leaves = jax.tree_util.tree_leaves(loaded_state_jax['opt_state'])
      for original_leaf, loaded_leaf in zip(original_leaves, loaded_leaves):
        if isinstance(original_leaf, (jnp.ndarray, jax.Array)):
          self.assertTrue(jnp.allclose(original_leaf, loaded_leaf))
        else:
          self.assertEqual(original_leaf, loaded_leaf)

  def test_load_non_existent_checkpoint(self):
    with self.assertRaises(FileNotFoundError):
      torchax.load_checkpoint('/path/to/non_existent_checkpoint')


if __name__ == '__main__':
  unittest.main()
