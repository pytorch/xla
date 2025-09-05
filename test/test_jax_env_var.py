import os
from absl.testing import absltest
from unittest.mock import patch
from torch_xla._internal.jax_workarounds import maybe_get_jax


class TestJaxEnvVar(absltest.TestCase):

  def setUp(self):
    # Clean up environment
    if 'TORCH_XLA_ENABLE_JAX' in os.environ:
      del os.environ['TORCH_XLA_ENABLE_JAX']

  def test_jax_enabled_attempts_import(self):
    os.environ['TORCH_XLA_ENABLE_JAX'] = '1'
    with patch(
        'torch_xla._internal.jax_workarounds.logging.warning') as mock_warn:
      result = maybe_get_jax()
      self.assertIsNone(result)
      mock_warn.assert_called_once()
      self.assertIn('JAX explicitly enabled but not installed',
                    mock_warn.call_args[0][0])

  def test_jax_unset_warns_with_guidance(self):
    """Test that unset environment variable warns with guidance"""
    with patch(
        'torch_xla._internal.jax_workarounds.logging.warning') as mock_warn:
      result = maybe_get_jax()
      self.assertIsNone(result)
      mock_warn.assert_called_once()
      warning_msg = mock_warn.call_args[0][0]
      self.assertIn('You are trying to use a feature that requires JAX',
                    warning_msg)
      self.assertIn('TORCH_XLA_ENABLE_JAX=1', warning_msg)
      self.assertIn('TORCH_XLA_ENABLE_JAX=0', warning_msg)

  def test_jax_disabled_values_silent(self):
    for val in ['0']:
      with self.subTest(val=val):
        os.environ['TORCH_XLA_ENABLE_JAX'] = val
        with patch(
            'torch_xla._internal.jax_workarounds.logging.warning') as mock_warn:
          result = maybe_get_jax()
          self.assertIsNone(result)
          mock_warn.assert_not_called()

  def test_jax_enabled_values(self):
    """Test various values that enable JAX"""
    for val in ['1']:
      with self.subTest(val=val):
        os.environ['TORCH_XLA_ENABLE_JAX'] = val
        with patch(
            'torch_xla._internal.jax_workarounds.logging.warning') as mock_warn:
          result = maybe_get_jax()
          self.assertIsNone(result)
          mock_warn.assert_called_once()
          self.assertIn('JAX explicitly enabled but not installed',
                        mock_warn.call_args[0][0])


if __name__ == '__main__':
  absltest.main()
