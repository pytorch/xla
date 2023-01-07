import torch
import torch.utils.data
import numpy as np

import contextlib
import gc
import io
import inspect
import itertools
import math
import random
import re
import copy
import os
import tempfile
import unittest
import warnings
import types
import pickle
import textwrap
import subprocess
import weakref
import sys
from torch._six import inf, nan, string_classes
from itertools import product, combinations, permutations
from functools import partial
from torch import multiprocessing as mp
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TEST_WITH_TORCHINDUCTOR, TestCase, TEST_WITH_ROCM, run_tests,
    IS_WINDOWS, IS_FILESYSTEM_UTF8_ENCODING, NO_MULTIPROCESSING_SPAWN,
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, load_tests, skipIfTorchInductor, slowTest,
    TEST_WITH_CROSSREF, skipIfTorchDynamo,
    skipCUDAMemoryLeakCheckIf, BytesIOContext,
    skipIfRocm, skipIfNoSciPy, TemporaryFileName, TemporaryDirectoryName,
    wrapDeterministicFlagAPITest, DeterministicGuard, CudaSyncGuard,
    skipIfNotRegistered, bytes_to_scalar, parametrize, skipIfMps, noncontiguous_like)
from multiprocessing.reduction import ForkingPickler
from torch.testing._internal.common_device_type import (
    expectedFailureMeta,
    expectedFailureXLA,
    instantiate_device_type_tests,
    onlyCUDA, onlyCPU,
    dtypes, dtypesIfCUDA, dtypesIfCPU, deviceCountAtLeast,
    skipMeta,
    PYTORCH_CUDA_MEMCHECK, largeTensorTest, onlyNativeDeviceTypes,
    get_all_device_types, skipXLA)
from typing import Tuple
import torch.backends.quantized
import torch.testing._internal.data
from torch.testing._internal.common_cuda import (
    tf32_on_and_off, tf32_is_not_fp32, TEST_CUDNN)
from torch.testing._internal.common_dtype import (
    floating_types_and, get_all_math_dtypes, all_types_and_complex_and, complex_types,
    all_types_and, floating_types, floating_and_complex_types,
)

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()


def test_cummax_cummin(self, device):
    def test_ops(op, string_of_function_name, expected_output1, expected_output2):
        x = torch.rand(100, 100, device=device)
        out1 = op(x, 1)
        res2 = torch.empty(0, device=device)
        indices2 = torch.empty(0, dtype=torch.int64, device=device)
        op(x, 1, out=(res2, indices2))
        self.assertEqual(out1[0], res2)
        self.assertEqual(out1[1], indices2)

        a = torch.tensor([[True, False, True],
                          [False, False, False],
                          [True, True, True]], dtype=torch.bool, device=device)
        b = a.byte()
        aRes = op(a, 0)
        bRes = op(b, 0)
        self.assertEqual(aRes[0], bRes[0].bool())
        self.assertEqual(aRes[0], expected_output1.bool())

        # test inf and nan input
        x = torch.tensor([4, inf, 1.5, -inf, 0, nan, 1])
        xRes = op(x, 0)[0]
        self.assertEqual(xRes, expected_output2)

        # op shouldn't support values, indices with a dtype, device type or layout
        # different from that of input tensor
        t = torch.randn(10)
        values = torch.empty(0, dtype=torch.int16)
        indices = torch.empty(0, dtype=torch.int64)
        with self.assertRaisesRegex(
                RuntimeError,
                'expected scalar_type Float but found Short'):
            op(t, 0, out=(values, indices))

        # Check that op over a zero length dimension doesn't crash on backprop.
        # Also check that op over other dimensions in a tensor with a zero-length
        # dimension also works
        # Also include a basic suite of similar tests for other bases cases.
        shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
        for shape in shapes:
            for dim in range(len(shape)):
                raw_tensor = torch.zeros(*shape, requires_grad=True)
                integrated = getattr(raw_tensor, string_of_function_name)(dim=dim)
                # Check that backward does not crash
                integrated[0].sum().backward()
                # Check that output maintained correct shape
                self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        # Check a scalar example
        raw_tensor = torch.tensor(3., requires_grad=True)
        integrated = getattr(raw_tensor, string_of_function_name)(dim=-1)
        # Check that backward does not crash
        integrated[0].sum().backward()
        # Check that output maintained correct shape
        self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

    expected_out = torch.tensor([4, inf, inf, inf, inf, nan, nan])
    test_ops(torch.cummax, "cummax", torch.tensor([[1, 0, 1],
                                                   [1, 0, 1],
                                                   [1, 1, 1]]), expected_out)

    expected_out = torch.tensor([4, 4, 1.5, -inf, -inf, nan, nan])
    test_ops(torch.cummin, "cummin", torch.tensor([[1, 0, 1],
                                                   [0, 0, 0],
                                                   [0, 0, 0]]), expected_out)



# The following block extends TestTorch with negative dim wrapping tests
# FIXME: replace these with OpInfo sample inputs or systemic OpInfo tests
# Functions to test negative dimension wrapping
METHOD = 1
INPLACE_METHOD = 2
FUNCTIONAL = 4
DIM_ARG = None

def make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim=0):
    def neg_dim_test(self):
        if isinstance(tensor_arg, list):
            assert METHOD not in types and INPLACE_METHOD not in types
            x = [torch.randn(arg) for arg in tensor_arg]
            ndim = len(tensor_arg[-1])
        else:
            x = torch.randn(*tensor_arg)
            ndim = len(tensor_arg)
        ndim += extra_dim

        n_dim_to_test = sum(e is DIM_ARG for e in arg_constr())

        for dims_val in combinations(range(ndim), n_dim_to_test):
            arg = arg_constr()
            arg_neg = copy.deepcopy(arg)
            idx = 0
            for i, v in enumerate(arg):
                if v is DIM_ARG:
                    arg[i] = dims_val[idx]
                    arg_neg[i] = dims_val[idx] - ndim
                    idx += 1

            if METHOD in types:
                a = getattr(x, name)(*arg)
                b = getattr(x, name)(*arg_neg)
                self.assertEqual(a, b)

            if INPLACE_METHOD in types:
                a = x.clone()
                getattr(a, name + '_')(*arg)
                b = x.clone()
                getattr(b, name + '_')(*arg_neg)
                self.assertEqual(a, b)

            if FUNCTIONAL in types:
                a = getattr(torch, name)(x, *arg)
                b = getattr(torch, name)(x, *arg_neg)
                self.assertEqual(a, b)

    return neg_dim_test

def idx_tensor(size, max_val):
    return torch.LongTensor(*size).random_(0, max_val - 1)

def add_neg_dim_tests():
    neg_dim_tests = [
#        ('narrow', (10, 20, 30), lambda: [DIM_ARG, 0, 5], [METHOD]),
#        ('transpose', (10, 20, 30), lambda: [DIM_ARG, DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
#        ('size', (10, 20, 30), lambda: [DIM_ARG], [METHOD]),
#        ('cat', [(2, 3, 4), (2, 3, 4)], lambda: [DIM_ARG], [FUNCTIONAL]),
#        ('chunk', (10, 20, 30), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('gather', (10, 20), lambda: [DIM_ARG, idx_tensor((10, 20), 10)], [METHOD, FUNCTIONAL]),
#        ('index_select', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10)], [METHOD, FUNCTIONAL]),
#        ('split', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('squeeze', (10, 1, 20, 1), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
#        ('unbind', (2, 3, 4), lambda: [DIM_ARG], [FUNCTIONAL]),
#        ('unsqueeze', (10, 20), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL], 1),
#        ('logcumsumexp', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('cumprod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('cumsum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('cummax', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('cummin', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('mean', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('median', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('nanmedian', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('mode', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('norm', (10, 20), lambda: [2, DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('prod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('std', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('sum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('var', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('kthvalue', (10, 20), lambda: [3, DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('max', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('min', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('sort', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('topk', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
#        ('renorm', (10, 20), lambda: [2, DIM_ARG, 1], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
#        ('index_add', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
#        ('index_copy', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
#        ('index_fill', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), 12], [INPLACE_METHOD]),
#        ('scatter', (10, 10), lambda: [DIM_ARG, idx_tensor((10, 10), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
#        ('select', (10, 20), lambda: [DIM_ARG, 3], [METHOD]),
#        ('unfold', (10, 20), lambda: [DIM_ARG, 5, 2], [METHOD]),
    ]

    for decl in neg_dim_tests:
        if len(decl) == 4:
            name, tensor_arg, arg_constr, types = decl
            extra_dim = 0
        elif len(decl) == 5:
            name, tensor_arg, arg_constr, types, extra_dim = decl

        test_name = 'test_' + name + '_neg_dim'

#        assert not hasattr(TestTorch, test_name), "Duplicated test name: " + test_name
        setattr(TestTorch, test_name, make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim))

# TODO: these empy classes are temporarily instantiated for XLA compatibility
#   once XLA updates their test suite it should be removed
class TestViewOps(TestCase):
    pass

class TestTensorDeviceOps(TestCase):
    pass

# Generates tests
# Note: test generation must be done at file scope, not within main, or
# pytest will fail.
add_neg_dim_tests()
#instantiate_device_type_tests(TestViewOps, globals())
#instantiate_device_type_tests(TestVitalSignsCuda, globals())
#instantiate_device_type_tests(TestTensorDeviceOps, globals())
instantiate_device_type_tests(TestTorchDeviceType, globals())
#instantiate_device_type_tests(TestDevicePrecision, globals(), except_for='cpu')


if __name__ == '__main__':
    run_tests()
