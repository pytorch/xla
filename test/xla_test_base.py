import copy
import os
import sys
from runpy import run_path

import torch_xla
import torch_xla.core.xla_model as xm

XLA_TEST_PATH = os.environ['XLA_TEST_DIR']
XLA_META = run_path(os.path.join(XLA_TEST_PATH, 'torch_test_meta.py'))
assert XLA_META, 'XLA metadata not found!'
DISABLED_TORCH_TESTS = XLA_META.get('disabled_torch_tests', None)
TORCH_TEST_PRECIIONS = XLA_META.get('torch_test_precisions', None)
DEFAULT_FLOATING_PRECISION = XLA_META.get('DEFAULT_FLOATING_PRECISION', None)
assert DISABLED_TORCH_TESTS is not None, 'XLA tests not found!'
assert TORCH_TEST_PRECIIONS is not None, 'XLA test precisions not found!'
assert DEFAULT_FLOATING_PRECISION is not None, ('DEFAULT_FLOATING_PRECISION not'
                                                ' found!')


class XLATestBase(DeviceTypeTestBase):
  device_type = 'xla'
  unsupported_dtypes = {torch.half, torch.complex64, torch.complex128}
  precision = DEFAULT_FLOATING_PRECISION

  @staticmethod
  def _alt_lookup(d, keys, defval):
    for k in keys:
      value = d.get(k, None)
      if value is not None:
        return value
    return defval

  # Overrides to instantiate tests that are known to run quickly
  # and correctly on XLA.
  @classmethod
  def instantiate_test(cls, name, test):
    test_name = name + '_' + cls.device_type

    @wraps(test)
    def disallowed_test(self, test=test):
      raise unittest.SkipTest('skipped on XLA')
      return test(self, cls.device_type)

    if test_name in DISABLED_TORCH_TESTS or test.__name__ in DISABLED_TORCH_TESTS:
      assert not hasattr(
          cls, test_name), 'Redefinition of test {0}'.format(test_name)
      setattr(cls, test_name, disallowed_test)
    else:  # Test is allowed
      dtypes = cls._get_dtypes(test)
      if dtypes is None:  # Tests without dtype variants are instantiated as usual
        super().instantiate_test(name, test)
      else:  # Tests with dtype variants have unsupported dtypes skipped
        # Sets default precision for floating types to bfloat16 precision
        if not hasattr(test, 'precision_overrides'):
          test.precision_overrides = {}
        xla_dtypes = []
        for dtype in dtypes:
          dtype_str = str(dtype).split('.')[1]
          dtype_test_name = test_name + '_' + dtype_str
          if dtype in cls.unsupported_dtypes:
            reason = 'XLA does not support dtype {0}'.format(str(dtype))

            @wraps(test)
            def skipped_test(self, *args, reason=reason, **kwargs):
              raise unittest.SkipTest(reason)

            assert not hasattr(
                cls, dtype_test_name), 'Redefinition of test {0}'.format(
                    dtype_test_name)
            setattr(cls, dtype_test_name, skipped_test)
          elif dtype_test_name in DISABLED_TORCH_TESTS:
            setattr(cls, dtype_test_name, disallowed_test)
          else:
            xla_dtypes.append(dtype)
          if dtype in [torch.float, torch.double, torch.bfloat16]:
            floating_precision = XLATestBase._alt_lookup(
                TORCH_TEST_PRECIIONS,
                [dtype_test_name, test_name, test.__name__],
                DEFAULT_FLOATING_PRECISION)
            test.precision_overrides[dtype] = floating_precision

        if len(xla_dtypes) != 0:
          test.dtypes[cls.device_type] = xla_dtypes
          super().instantiate_test(name, test)

  @classmethod
  def get_primary_device(cls):
    return cls.primary_device

  @classmethod
  def setUpClass(cls):
    # Sets the primary test device to the xla_device (CPU or TPU)
    cls.primary_device = str(xm.xla_device())
    torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
        use_full_mat_mul_precision=True)

  # Overrides assertEqual to popular custom precision
  def assertEqual(self, x, y, prec=None, message='', allow_inf=False, **kwargs):
    if prec is None:
      prec = self.precision
    else:
      prec = max(self.precision, prec)
    gmode = os.environ.get('TEST_PRINT_GRAPH', '').lower()
    if gmode == 'text':
      if type(x) == torch.Tensor and xm.is_xla_tensor(x):
        print(
            '\nTest Graph (x):\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_text([x])),
            file=sys.stderr)
      if type(y) == torch.Tensor and xm.is_xla_tensor(y):
        print(
            '\nTest Graph (y):\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_text([y])),
            file=sys.stderr)
    elif gmode == 'hlo':
      if type(x) == torch.Tensor and xm.is_xla_tensor(x):
        print(
            '\nTest Graph (x):\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_hlo([x])),
            file=sys.stderr)
      if type(y) == torch.Tensor and xm.is_xla_tensor(y):
        print(
            '\nTest Graph (y):\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_hlo([y])),
            file=sys.stderr)
    elif gmode:
      raise RuntimeError('Invalid TEST_PRINT_GRAPH value: {}'.format(gmode))
    if type(x) == torch.Tensor:
      x = x.cpu()
    if type(y) == torch.Tensor:
      y = y.cpu()
    return DeviceTypeTestBase.assertEqual(self, x, y, prec, message, allow_inf,
                                          **kwargs)
