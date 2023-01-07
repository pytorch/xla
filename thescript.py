# -*- coding: utf-8 -*-
# Owner(s): ["module: tests"]

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

@contextlib.contextmanager
def torch_vital_set(value):
    stash = None
    if 'TORCH_VITAL' in os.environ:
        stash = os.environ['TORCH_VITAL']
    os.environ['TORCH_VITAL'] = value
    try:
        yield
    finally:
        if stash:
            os.environ['TORCH_VITAL'] = stash
        else:
            del os.environ['TORCH_VITAL']


is_cuda_sm86 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 6)



class TestTorchDeviceType(TestCase):
    exact_dtype = True

    # TODO: move all tensor creation to common ops
    def _rand_shape(self, dim, min_size, max_size):
        shape = []
        for i in range(dim):
            shape.append(random.randint(min_size, max_size))
        return tuple(shape)

    @skipIfMps
    def test_cummax_cummin(self, device):
        import pdb; pdb.set_trace()
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

        print("before the try one1111111111111111111111111111111111111111111111")
        expected_out = torch.tensor([4, inf, inf, inf, inf, nan, nan])
        print("before the test_ops func2222222222222222222222222222222222222222")
        test_ops(torch.cummax, "cummax", torch.tensor([[1, 0, 1],
                                                       [1, 0, 1],
                                                       [1, 1, 1]]), expected_out)
        print("after the test_ops func33333333333333333333333333333333333333333")

        print("before the try two4444444444444444444444444444444444444444444444")
        expected_out = torch.tensor([4, 4, 1.5, -inf, -inf, nan, nan])
        print("before the test_ops func5555555555555555555555555555555555555555")
        test_ops(torch.cummin, "cummin", torch.tensor([[1, 0, 1],
                                                       [0, 0, 0],
                                                       [0, 0, 0]]), expected_out)
        print("after the test_ops func66666666666666666666666666666666666666666")



# we implemented custom deallocation for subclasses, so it behooves
# us to make sure all of these bits work.  We'll use __del__ to
# track if objects die or not
class Tracker:
    def __init__(self, marker):
        self.marker = marker

    @staticmethod
    def make():
        marker = [False]
        return marker, Tracker(marker)

    def __del__(self):
        self.marker[0] = True

@contextlib.contextmanager
def disable_gc():
    if gc.isenabled():
        try:
            gc.disable()
            yield
        finally:
            gc.enable()
    else:
        yield

class TestTorch(TestCase):
    exact_dtype = True

    def test_dir(self):
        dir(torch)

    def test_wildcard_import(self):
        exec('from torch import *')

    def test_newaxis_numpy_comparison(self):
        def run_test(tensor, *idx):
            npt = tensor.numpy()
            self.assertEqual(tensor[idx], npt[idx])

        # 1D Tensor Tests
        x = torch.arange(0, 10)
        cases = [
            [None],
            [None, None],
            [Ellipsis, None],
            [None, Ellipsis],
            [2, None],
            [None, 2],
            [Ellipsis, None, 2],
            [Ellipsis, 2, None],
            [2, Ellipsis, None],
            [2, None, Ellipsis],
            [None, 2, Ellipsis],
            [None, Ellipsis, 2],
        ]

        for case in cases:
            run_test(x, *case)

        # 2D Tensor Tests
        x = torch.arange(0, 12).view(3, 4)
        cases = [
            [None],
            [None, None],
            [None, None, None],
            [Ellipsis, None],
            [Ellipsis, None, None],
            [None, Ellipsis],
            [None, Ellipsis, None],
            [None, None, Ellipsis],
            [2, None],
            [2, None, Ellipsis],
            [2, Ellipsis, None],
            [None, 2, Ellipsis],
            [Ellipsis, 2, None],
            [Ellipsis, None, 2],
            [None, Ellipsis, 2],
            [1, 2, None],
            [1, 2, Ellipsis, None],
            [1, Ellipsis, 2, None],
            [Ellipsis, 1, None, 2],
            [Ellipsis, 1, 2, None],
            [1, None, 2, Ellipsis],
            [None, 1, Ellipsis, 2],
            [None, 1, 2, Ellipsis],
        ]

        for case in cases:
            run_test(x, *case)

    def _consecutive(self, size, start=1):
        sequence = torch.ones(torch.tensor(size).prod(0)).cumsum(0)
        sequence.add_(start - 1)
        return sequence.resize_(*size)

    def test_newindex(self):
        reference = self._consecutive((3, 3, 3))
        # This relies on __index__() being correct - but we have separate tests for that

        def checkPartialAssign(index):
            reference = torch.zeros(3, 3, 3)
            reference[index] = self._consecutive((3, 3, 3))[index]
            self.assertEqual(reference[index], self._consecutive((3, 3, 3))[index], atol=0, rtol=0)
            reference[index] = 0
            self.assertEqual(reference, torch.zeros(3, 3, 3), atol=0, rtol=0)

        checkPartialAssign(0)
        checkPartialAssign(1)
        checkPartialAssign(2)
        checkPartialAssign((0, 1))
        checkPartialAssign((1, 2))
        checkPartialAssign((0, 2))
        checkPartialAssign(torch.LongTensor((0, 2)))

        with self.assertRaises(IndexError):
            reference[1, 1, 1, 1] = 1
        with self.assertRaises(IndexError):
            reference[1, 1, 1, (1, 1)] = 1
        with self.assertRaises(IndexError):
            reference[3, 3, 3, 3, 3, 3, 3, 3] = 1
        with self.assertRaises(IndexError):
            reference[0.0] = 1
        with self.assertRaises(TypeError):
            reference[0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, :, 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, ..., 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, :, 0.0] = 1

    # FIXME: move to indexing test suite
    def test_index_add(self):
        for device in get_all_device_types():
            for dest_contig, src_contig, index_contig in product([True, False], repeat=3):
                for other_sizes in ((), (4, 5)):
                    for dtype in [torch.int, torch.long]:
                        num_copy, num_dest = 3, 3
                        dest = torch.randn(num_dest, *other_sizes, device=device)
                        if not dest_contig:
                            dest = make_tensor(dest.shape, device=device, dtype=dest.dtype, noncontiguous=True)
                        src = torch.randn(num_copy, *other_sizes, device=device)
                        if not src_contig:
                            src = noncontiguous_like(src)
                        idx = torch.randperm(num_dest, dtype=dtype, device=device).narrow(0, 0, num_copy)
                        if not index_contig:
                            idx = noncontiguous_like(idx)
                        # index_add_ without alpha argument
                        dest2 = dest.clone()
                        dest.index_add_(0, idx, src)
                        for i in range(idx.size(0)):
                            dest2[idx[i]] += src[i]
                        self.assertEqual(dest, dest2)
                        # index_add_ with alpha argument
                        dest2 = dest.clone()
                        dest.index_add_(0, idx, src, alpha=2)
                        for i in range(idx.size(0)):
                            dest2[idx[i]] += src[i] * 2
                        self.assertEqual(dest, dest2)

    # FIXME: resolve comment below and move this to indexing test suite
    # add coverage for issue with atomic add that appeared only for
    # specific dtypes on cuda:
    # https://github.com/pytorch/pytorch/issues/29153
    def test_index_add_all_dtypes(self):
        for device in get_all_device_types():
            for dtype in get_all_math_dtypes(device):
                for idx_dtype in [torch.int, torch.long]:
                    size = [5, 5]
                    if dtype.is_floating_point or dtype.is_complex:
                        tensor = torch.rand(size, dtype=dtype, device=device)
                    elif dtype.is_signed:
                        tensor = torch.randint(-5, 15, size, dtype=dtype, device=device)
                    else:
                        tensor = torch.randint(0, 10, size, dtype=dtype, device=device)

                    # index_add calls atomicAdd on cuda.
                    zeros = torch.zeros(size, dtype=dtype, device=device)

                    added = zeros.index_add(0, torch.arange(0, size[0], dtype=idx_dtype, device=device), tensor)
                    self.assertEqual(added, tensor)

                    added = zeros.index_add(0, torch.arange(0, size[0], dtype=idx_dtype, device=device), tensor, alpha=-1)
                    self.assertEqual(added, -tensor)

    def test_index_add_correctness(self):
        # Check whether index_add can get correct result when
        # alpha is 1, and dtype of index is torch.long,
        # i.e., using scatter_add
        def helper(dim, dtype, device, size_result, size_source):
            tensor = torch.zeros(size_result, dtype=dtype, device=device)
            index = torch.randint(0, size_result[dim], (size_source[dim],),
                                  dtype=torch.long, device=device)
            if dtype.is_floating_point or dtype.is_complex:
                source = torch.rand(size_source, dtype=dtype, device=device)
            elif dtype.is_signed:
                source = torch.randint(-2, 5, size_source, dtype=dtype, device=device)
            else:
                source = torch.randint(0, 5, size_source, dtype=dtype, device=device)

            ref_out = tensor.index_add(dim, index, source, alpha=2.) / 2.
            ref_out = ref_out.to(dtype=dtype)
            out = tensor.index_add(dim, index, source)
            if device == 'cuda':
                self.assertEqual(out, ref_out, atol=1e-2, rtol=1e-2)
            else:
                self.assertEqual(out, ref_out.to(dtype=dtype))

        for dim in [-1, -2, -3]:
            for dtype in all_types_and_complex_and(torch.half, torch.bfloat16):
                for device in get_all_device_types():
                    for size in [(2, 512, 256), (5, 256, 256)]:
                        helper(dim, dtype, device, size, size)

                # Check broadcast cases on CPU
                size_result = (2, 512, 256)
                size_source = (1, 512, 256)
                helper(dim, dtype, 'cpu', size_result, size_source)
                size_result = (2, 512, 512)
                size_source = (1, 512, 1)
                helper(dim, dtype, 'cpu', size_result, size_source)
                size_result = (2, 512, 256)
                size_source = (2, 1, 256)
                helper(dim, dtype, 'cpu', size_result, size_source)

                # Check bound
                result = torch.zeros(1, 512, 256, dtype=dtype)
                source = torch.ones(1, 512, 256, dtype=dtype)
                index = torch.ones(257).to(dtype=torch.long)
                self.assertRaises(RuntimeError, lambda: result.index_add_(dim, index, source))
                index = (torch.ones(256) * 257).to(dtype=torch.long)
                self.assertRaises(RuntimeError, lambda: result.index_add_(dim, index, source))

    # FIXME: move to shape ops test suite
    def test_unflatten(self):
        # test args: tensor, int, sizes
        self.assertEqual(torch.tensor([]).unflatten(0, (0, 1)), torch.empty(0, 1))
        self.assertEqual(torch.tensor([1]).unflatten(0, (1, 1)), torch.tensor([[1]]))
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, (2, 2)), torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, [2, 2]), torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, torch.Size([2, 2])), torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(torch.ones(2, 10).unflatten(1, (5, 2)), torch.ones(2, 5, 2))
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, (-1, 2)),
                         torch.tensor([[1, 2], [3, 4]]))
        self.assertEqual(torch.ones(2, 10).unflatten(1, (5, -1)),
                         torch.ones(2, 5, 2))
        self.assertEqual(torch.ones(2, 10).unflatten(1, (-1,)),
                         torch.ones(2, 10))
        self.assertEqual(torch.ones(2, 3 * 4 * 5 * 6).unflatten(1, (3, 4, -1, 6)),
                         torch.ones(2, 3, 4, 5, 6))
        self.assertEqual(torch.ones(2, 0, 2).unflatten(1, (3, -1, 4, 5)),
                         torch.ones(2, 3, 0, 4, 5, 2))

        # test invalid args: tensor, str, sizes
        with self.assertRaisesRegex(TypeError, r"unflatten\(\): argument 'dim' \(position 1\) must be int, not str"):
            torch.tensor([1]).unflatten('A', (1, 1))

        # test invalid args: tensor, str, namedshape
        with self.assertRaisesRegex(RuntimeError, r"Name 'A' not found in Tensor\[None\]."):
            torch.ones(4).unflatten('A', (('A', 2), ('B', 2)))

        # test other invalid arguments
        with self.assertRaisesRegex(RuntimeError, r"sizes must be non-empty"):
            torch.tensor([1]).unflatten(0, [])
        with self.assertRaisesRegex(RuntimeError, r"Provided sizes \[2, 2\] don't multiply up to the size of dim 0 \(1\)"):
            torch.tensor([1]).unflatten(0, [2, 2])
        with self.assertRaisesRegex(IndexError, r"Dimension specified as 0 but tensor has no dimensions"):
            torch.tensor(1).unflatten(0, [0])
        with self.assertRaisesRegex(RuntimeError, r"only one dimension can be inferred"):
            torch.randn(5, 10).unflatten(1, (-1, -1))
        with self.assertRaisesRegex(RuntimeError,
                                    r"Provided sizes \[-1, 4\] don't multiply up to the size of dim 1 \(10\)"):
            torch.randn(5, 10).unflatten(1, (-1, 4))
        with self.assertRaisesRegex(RuntimeError,
                                    r"the unspecified dimension size -1 can be any value and is ambiguous"):
            torch.randn(2, 0).unflatten(1, (2, -1, 0))

    # Test that warnings generated from C++ are translated to the correct type
    def test_warn_types(self):
        test_cases = [
            # function, warning type, message
            (torch._C._warn, UserWarning, r"Test message for TORCH_WARN"),
            (torch._C._warn_deprecation, DeprecationWarning, r"Test message for TORCH_WARN_DEPRECATION"),
        ]

        for fn, warning_type, message in test_cases:
            with warnings.catch_warnings(record=True) as w:
                warnings.resetwarnings()
                warnings.filterwarnings('always', category=warning_type)
                fn()

                self.assertEqual(len(w), 1, msg=f'{warning_type} not raised')
                warning = w[0].message
                self.assertTrue(isinstance(warning, warning_type), msg=f'{warning_type} not raised')
                self.assertTrue(re.search(
                    message,
                    str(warning)))

    def test_structseq_repr(self):
        a = torch.arange(250).reshape(5, 5, 10)
        expected = """
        torch.return_types.max(
        values=tensor([[ 40,  41,  42,  43,  44,  45,  46,  47,  48,  49],
                [ 90,  91,  92,  93,  94,  95,  96,  97,  98,  99],
                [140, 141, 142, 143, 144, 145, 146, 147, 148, 149],
                [190, 191, 192, 193, 194, 195, 196, 197, 198, 199],
                [240, 241, 242, 243, 244, 245, 246, 247, 248, 249]]),
        indices=tensor([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]))"""
        self.assertEqual(repr(a.max(1)), textwrap.dedent(expected).strip())

    def test_is_same_size(self):
        t1 = torch.empty(3, 4, 9, 10)
        t2 = torch.empty(3, 4)
        t3 = torch.empty(1, 9, 3, 3)
        t4 = torch.empty(3, 4, 9, 10)

        self.assertFalse(t1.is_same_size(t2))
        self.assertFalse(t1.is_same_size(t3))
        self.assertTrue(t1.is_same_size(t4))

        nt1 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(3, 4), torch.ones(5, 4)])
        nt2 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4)])
        nt3 = torch.nested.nested_tensor([torch.ones(2, 4, 5), torch.ones(2, 6, 5)])
        nt4 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(3, 4), torch.ones(5, 4)])

        self.assertFalse(nt1.is_same_size(nt2))
        self.assertFalse(nt1.is_same_size(nt3))
        self.assertTrue(nt1.is_same_size(nt4))
        with self.assertRaisesRegex(RuntimeError, "Expected both self and other to be nested tensors."):
            t1.is_same_size(nt1)

        with self.assertRaisesRegex(RuntimeError, "Expected both self and other to be nested tensors."):
            nt1.is_same_size(t1)

    def test_tensor_set(self):
        t1 = torch.tensor([])
        t2 = torch.empty(3, 4, 9, 10).uniform_()
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        size = torch.Size([9, 3, 4, 10])
        t1.set_(t2.storage(), 0, size)
        self.assertEqual(t1.size(), size)
        t1.set_(t2.storage(), 0, tuple(size))
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), (120, 40, 10, 1))
        stride = (10, 360, 90, 1)
        t1.set_(t2.storage(), 0, size, stride)
        self.assertEqual(t1.stride(), stride)
        t1.set_(t2.storage(), 0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        # test argument names
        t1 = torch.tensor([])
        # 1. case when source is tensor
        t1.set_(source=t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 2. case when source is storage
        t1.set_(source=t2.storage())
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 3. case when source is storage, and other args also specified
        t1.set_(source=t2.storage(), storage_offset=0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        t1 = torch.tensor([True, True], dtype=torch.bool)
        t2 = torch.tensor([False, False], dtype=torch.bool)
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)

    def test_tensor_set_errors(self):
        f_cpu = torch.randn((2, 3), dtype=torch.float32)
        d_cpu = torch.randn((2, 3), dtype=torch.float64)

        # change dtype
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu.storage()))
        self.assertRaises(RuntimeError,
                          lambda: f_cpu.set_(d_cpu.storage(), 0, d_cpu.size(), d_cpu.stride()))
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu))

        # change device
        if torch.cuda.is_available():
            f_cuda = torch.randn((2, 3), dtype=torch.float32, device='cuda')

            # cpu -> cuda
            self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_cuda.storage()))
            self.assertRaises(RuntimeError,
                              lambda: f_cpu.set_(f_cuda.storage(), 0, f_cuda.size(), f_cuda.stride()))
            self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_cuda))

            # cuda -> cpu
            self.assertRaises(RuntimeError, lambda: f_cuda.set_(f_cpu.storage()))
            self.assertRaises(RuntimeError,
                              lambda: f_cuda.set_(f_cpu.storage(), 0, f_cpu.size(), f_cpu.stride()))
            self.assertRaises(RuntimeError, lambda: f_cuda.set_(f_cpu))

    # FIXME: move this test test_testing.py (along with allclose testing)
    # NOTE: test_equal will be deprecated in favor of torch.testing.assert_close
    #   once torch.testing is out of beta
    def test_equal(self):
        # Contiguous, 1D
        t1 = torch.tensor((3., 4., 9., 10.))
        t2 = t1.contiguous()
        t3 = torch.tensor((1., 9., 3., 10.))
        t4 = torch.tensor((3., 4., 9.))
        t5 = torch.tensor([])
        self.assertTrue(t1.equal(t2))
        self.assertFalse(t1.equal(t3))
        self.assertFalse(t1.equal(t4))
        self.assertFalse(t1.equal(t5))
        self.assertTrue(torch.equal(t1, t2))
        self.assertFalse(torch.equal(t1, t3))
        self.assertFalse(torch.equal(t1, t4))
        self.assertFalse(torch.equal(t1, t5))

        # Non contiguous, 2D
        s = torch.tensor(((1, 2, 3, 4), (5, 6, 7, 8)))
        s1 = s[:, 1:3]
        s2 = s1.clone()
        s3 = torch.tensor(((2, 3), (6, 7)))
        s4 = torch.tensor(((0, 0), (0, 0)))

        self.assertFalse(s1.is_contiguous())
        self.assertTrue(s1.equal(s2))
        self.assertTrue(s1.equal(s3))
        self.assertFalse(s1.equal(s4))
        self.assertTrue(torch.equal(s1, s2))
        self.assertTrue(torch.equal(s1, s3))
        self.assertFalse(torch.equal(s1, s4))

        # Different dtypes
        x = torch.tensor((1, 2, 3), dtype=torch.float)
        y = torch.tensor((1, 2, 3), dtype=torch.int)
        z = torch.tensor((1, -1), dtype=torch.int)
        self.assertTrue(torch.equal(x, y))
        self.assertFalse(torch.equal(z, x))

    def test_element_size(self):
        byte = torch.ByteStorage().element_size()
        char = torch.CharStorage().element_size()
        short = torch.ShortStorage().element_size()
        int = torch.IntStorage().element_size()
        long = torch.LongStorage().element_size()
        float = torch.FloatStorage().element_size()
        double = torch.DoubleStorage().element_size()
        bool = torch.BoolStorage().element_size()
        bfloat16 = torch.BFloat16Storage().element_size()
        complexfloat = torch.ComplexFloatStorage().element_size()
        complexdouble = torch.ComplexDoubleStorage().element_size()

        self.assertEqual(byte, torch.ByteTensor().element_size())
        self.assertEqual(char, torch.CharTensor().element_size())
        self.assertEqual(short, torch.ShortTensor().element_size())
        self.assertEqual(int, torch.IntTensor().element_size())
        self.assertEqual(long, torch.LongTensor().element_size())
        self.assertEqual(float, torch.FloatTensor().element_size())
        self.assertEqual(double, torch.DoubleTensor().element_size())
        self.assertEqual(bool, torch.BoolTensor().element_size())
        self.assertEqual(bfloat16, torch.tensor([], dtype=torch.bfloat16).element_size())
        self.assertEqual(complexfloat, torch.tensor([], dtype=torch.complex64).element_size())
        self.assertEqual(complexdouble, torch.tensor([], dtype=torch.complex128).element_size())

        self.assertGreater(byte, 0)
        self.assertGreater(char, 0)
        self.assertGreater(short, 0)
        self.assertGreater(int, 0)
        self.assertGreater(long, 0)
        self.assertGreater(float, 0)
        self.assertGreater(double, 0)
        self.assertGreater(bool, 0)
        self.assertGreater(bfloat16, 0)
        self.assertGreater(complexfloat, 0)
        self.assertGreater(complexdouble, 0)

        # These tests are portable, not necessarily strict for your system.
        self.assertEqual(byte, 1)
        self.assertEqual(char, 1)
        self.assertEqual(bool, 1)
        self.assertGreaterEqual(short, 2)
        self.assertGreaterEqual(int, 2)
        self.assertGreaterEqual(int, short)
        self.assertGreaterEqual(long, 4)
        self.assertGreaterEqual(long, int)
        self.assertGreaterEqual(double, float)

    def test_permute(self):
        orig = [1, 2, 3, 4, 5, 6, 7]
        perm = torch.randperm(7).tolist()
        x = torch.empty(*orig).fill_(0)
        new = [i - 1 for i in x.permute(*perm).size()]
        self.assertEqual(perm, new)
        self.assertEqual(x.size(), orig)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_reversed(self):
        val = torch.arange(0, 10)
        self.assertEqual(reversed(val), torch.arange(9, -1, -1))

        val = torch.arange(1, 10).view(3, 3)
        self.assertEqual(reversed(val), torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]]))

        val = torch.tensor(42)
        self.assertEqual(reversed(val), torch.tensor(42))

    def test_contains(self):
        x = torch.arange(0, 10)
        self.assertEqual(4 in x, True)
        self.assertEqual(12 in x, False)

        x = torch.arange(1, 10).view(3, 3)
        val = torch.arange(1, 4)
        self.assertEqual(val in x, True)
        val += 10
        self.assertEqual(val in x, False)

        self.assertRaisesRegex(
            RuntimeError,
            "Tensor.__contains__ only supports Tensor or scalar, but you passed in a {}.".format(type("foo")),
            lambda: "foo" in x)
        self.assertRaisesRegex(
            RuntimeError,
            "Tensor.__contains__ only supports Tensor or scalar, but you passed in a {}.".format(type([1, 2])),
            lambda: [1, 2] in x)

    def test_deepcopy_parameter(self):
        from copy import deepcopy
        l = torch.nn.Linear(10, 1)
        s = l.state_dict(keep_vars=True)
        self.assertEqual(torch.nn.Parameter, type(s['weight']))
        self.assertEqual(torch.nn.Parameter, type(s['bias']))

        s2 = deepcopy(s)
        self.assertEqual(torch.nn.Parameter, type(s2['weight']))
        self.assertEqual(torch.nn.Parameter, type(s2['bias']))

    def test_pickle(self):
        import pickle
        a = torch.randn(5, 5)
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertEqual(a, b)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_pickle_parameter(self):
        import pickle
        a = torch.nn.Parameter(torch.randn(5, 5))
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertTrue(isinstance(b, torch.nn.Parameter))
        self.assertEqual(a.requires_grad, b.requires_grad)
        self.assertEqual(a, b)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_pickle_parameter_no_requires_grad(self):
        import pickle
        a = torch.nn.Parameter(torch.randn(5, 5), requires_grad=False)
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertTrue(isinstance(b, torch.nn.Parameter))
        self.assertEqual(a.requires_grad, b.requires_grad)
        self.assertEqual(a, b)

    def test_pickle_dtype(self):
        t = torch.float32
        serialized = pickle.dumps(t)
        b = pickle.loads(serialized)
        self.assertTrue(isinstance(b, torch.dtype))
        self.assertEqual(id(b), id(t))

    def test_pickle_size(self):
        a = torch.rand(10).size()
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertTrue(isinstance(b, torch.Size))
        self.assertEqual(a, b)

    def test_pickle_function(self):
        # https://github.com/pytorch/pytorch/issues/37703
        a = torch.tanh
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertEqual(a, b)

    def test_generator_cpu(self):
        # test default generators are equal
        self.assertEqual(torch.default_generator, torch.default_generator)

        # tests Generator API
        # manual_seed, seed, initial_seed, get_state, set_state
        g1 = torch.Generator()
        g2 = torch.Generator()
        g1.manual_seed(12345)
        g2.manual_seed(12345)
        self.assertEqual(g1.initial_seed(), g2.initial_seed())

        g1.seed()
        g2.seed()
        self.assertNotEqual(g1.initial_seed(), g2.initial_seed())

        g1 = torch.Generator()
        g2_state = g2.get_state()
        g2_randn = torch.randn(1, generator=g2)
        g1.set_state(g2_state)
        g1_randn = torch.randn(1, generator=g1)
        self.assertEqual(g1_randn, g2_randn)

        default_state = torch.default_generator.get_state()
        q = torch.empty(100)
        g1_normal = q.normal_()
        g2 = torch.Generator()
        g2.set_state(default_state)
        g2_normal = q.normal_(generator=g2)
        self.assertEqual(g1_normal, g2_normal)

    def test_invalid_generator_raises(self):
        self.assertRaises(RuntimeError, lambda: torch.Generator('opengl'))

    def _sobol_reference_samples(self, scramble: bool) -> torch.Tensor:
        if not scramble:
            # theoretical values from Joe Kuo 2010
            return torch.tensor(
                [
                    [0., 0.],
                    [0.5, 0.5],
                    [0.75, 0.25],
                    [0.25, 0.75],
                    [0.375, 0.375],
                    [0.875, 0.875],
                    [0.625, 0.125],
                    [0.125, 0.625],
                ],
            )
        else:
            # theoretical values unknown: convergence properties checked
            return torch.tensor(
                [
                    [0.50860737, 0.29320504],
                    [0.07116939, 0.89594537],
                    [0.49354145, 0.11524881],
                    [0.93097717, 0.70244044],
                    [0.87266153, 0.23887917],
                    [0.31021884, 0.57600391],
                    [0.13687253, 0.42054182],
                    [0.69931293, 0.77336788],
                ],
            )

    def test_sobolengine_bounds(self, scramble: bool = False):
        engine = torch.quasirandom.SobolEngine(100, scramble=scramble, seed=123456)
        sample = engine.draw(512)
        self.assertTrue(torch.all(sample >= 0))
        self.assertTrue(torch.all(sample <= 1))

    def test_sobolengine_bounds_scrambled(self):
        self.test_sobolengine_bounds(scramble=True)

    def test_sobolengine_draw(self, scramble: bool = False):
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        sample = engine.draw(n=len(ref_sample))
        self.assertEqual(sample, ref_sample)
        self.assertEqual(engine.num_generated, len(ref_sample))

    def test_sobolengine_draw_scrambled(self):
        self.test_sobolengine_draw(scramble=True)

    def test_sobolengine_first_point(self):
        for dtype in (torch.float, torch.double):
            engine = torch.quasirandom.SobolEngine(2, scramble=False)
            sample = engine.draw(1, dtype=dtype)
            self.assertTrue(torch.all(sample == 0))
            self.assertEqual(sample.dtype, dtype)
        for dtype in (torch.float, torch.double):
            engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=123456)
            sample = engine.draw(1, dtype=dtype)
            self.assertTrue(torch.all(sample != 0))
            self.assertEqual(sample.dtype, dtype)

    def test_sobolengine_continuing(self, scramble: bool = False):
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        n_half = len(ref_sample) // 2
        _ = engine.draw(n=n_half)
        sample = engine.draw(n=n_half)
        torch.testing.assert_close(sample, ref_sample[n_half:])

    def test_sobolengine_continuing_scrambled(self):
        self.test_sobolengine_continuing(scramble=True)

    def test_sobolengine_reset(self, scramble: bool = False):
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        _ = engine.draw(n=len(ref_sample) // 2)
        engine.reset()
        self.assertEqual(engine.num_generated, 0)
        sample = engine.draw(n=len(ref_sample))
        torch.testing.assert_close(sample, ref_sample)

    def test_sobolengine_reset_scrambled(self):
        self.test_sobolengine_reset(scramble=True)

    def test_sobolengine_fast_forward(self, scramble: bool = False):
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        engine.fast_forward(4)
        sample = engine.draw(n=4)
        torch.testing.assert_close(sample, ref_sample[4:])
        # alternate fast forwarding with sampling
        engine.reset()
        even_draws = []
        for i in range(8):
            if i % 2 == 0:
                even_draws.append(engine.draw())
            else:
                engine.fast_forward(1)
        torch.testing.assert_close(
            ref_sample[[i for i in range(8) if i % 2 == 0]],
            torch.from_numpy(np.concatenate(even_draws)),
        )

    def test_sobolengine_fast_forward_scrambled(self):
        self.test_sobolengine_fast_forward(scramble=True)

    def test_sobolengine_distribution(self, scramble=False):
        d = 50
        engine = torch.quasirandom.SobolEngine(d, scramble=scramble, seed=123456)
        sample = engine.draw(1024)
        torch.testing.assert_close(
            torch.mean(sample, dim=0), torch.full((d,), 0.5), atol=2, rtol=2
        )
        torch.testing.assert_close(
            np.percentile(sample, 25, axis=0), np.repeat(0.25, d), atol=2, rtol=2
        )
        torch.testing.assert_close(
            np.percentile(sample, 75, axis=0), np.repeat(0.75, d), atol=2, rtol=2
        )

    def test_sobolengine_distribution_scrambled(self):
        self.test_sobolengine_distribution(scramble=True)

    def test_sobolengine_draw_base2(self, scramble=False):
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        sample = engine.draw_base2(2)
        self.assertEqual(ref_sample[:4], sample)
        # resampling still having N=2**n
        sample = engine.draw_base2(2)
        self.assertEqual(ref_sample[4:8], sample)

    def test_sobolengine_draw_base2_scrambled(self):
        self.test_sobolengine_draw_base2(scramble=True)

    def test_sobolengine_raise(self):
        maxdim = torch.quasirandom.SobolEngine.MAXDIM
        with self.assertRaises(ValueError):
            torch.quasirandom.SobolEngine(maxdim + 1)

    def test_sobolengine_high_dim(self):
        engine = torch.quasirandom.SobolEngine(1111, scramble=False, seed=123456)
        samples1 = engine.draw()
        vals1, counts1 = torch.unique(samples1, return_counts=True)
        samples2 = engine.draw()
        vals2, counts2 = torch.unique(samples2, return_counts=True)
        self.assertEqual(vals1.item(), 0.0)
        self.assertEqual(counts1.item(), 1111)
        self.assertEqual(vals2.item(), 0.5)
        self.assertEqual(counts1.item(), 1111)

    def test_parsing_int64(self):
        # accepts integer arguments
        x = torch.cumsum(torch.ones(5, 5), 0)
        self.assertEqual(x, torch.cumsum(torch.ones(5, 5), torch.tensor(0)))
        # doesn't accept floating point variables
        self.assertRaises(TypeError, lambda: torch.cumsum(torch.ones(5, 5), torch.tensor(0.)))

    def test_parsing_double(self):
        # accepts floating point and integer arguments
        x = torch.randn(2, 3)
        torch.isclose(x, x, 1, 1)
        self.assertTrue(torch.isclose(x, x, 1, 1).all())
        self.assertTrue(torch.isclose(x, x, 1.5, 1.).all())
        # accepts floating point and integer tensors
        self.assertTrue(torch.isclose(x, x, torch.tensor(1), torch.tensor(1)).all())
        self.assertTrue(torch.isclose(x, x, torch.tensor(1.5), torch.tensor(1.)).all())
        # doesn't accept variables with requires_grad
        self.assertRaises(TypeError,
                          lambda: torch.isclose(x, x, torch.tensor(1.5), torch.tensor(1., requires_grad=True)).all())

    def test_parsing_intlist(self):
        #  parse with integer variables
        self.assertEqual(torch.Size([3, 4]), torch.ones((torch.tensor(3), torch.tensor(4))).shape)
        self.assertEqual(torch.Size([3, 4]), torch.ones(torch.tensor(3), torch.tensor(4)).shape)
        # parse with numpy integers
        self.assertEqual(torch.Size([3, 4]), torch.ones((np.array(3), np.int64(4))).shape)
        self.assertEqual(torch.Size([3, 4]), torch.ones(np.array(3), np.int64(4)).shape)
        self.assertEqual(torch.Size([3, 4]), torch.ones((np.int64(3), np.array(4))).shape)
        self.assertEqual(torch.Size([3, 4]), torch.ones(np.int64(3), np.array(4)).shape)

        # fail parse with float variables
        self.assertRaises(TypeError, lambda: torch.ones((torch.tensor(3.), torch.tensor(4))))
        # fail parse with numpy floats
        self.assertRaises(TypeError, lambda: torch.ones((np.float(3.), torch.tensor(4))))
        self.assertRaises(TypeError, lambda: torch.ones((np.array(3.), torch.tensor(4))))

        # fail parse with > 1 element variables
        self.assertRaises(TypeError, lambda: torch.ones(torch.tensor(3, 3)))
        self.assertRaises(TypeError, lambda: torch.ones((torch.tensor(3, 3))))
        self.assertRaises(TypeError, lambda: torch.ones(np.array(3, 3)))
        self.assertRaises(TypeError, lambda: torch.ones((np.array(3, 3))))

        # fail parse with additional positional args after intlist arg
        self.assertRaisesRegex(TypeError,
                               "received an invalid combination of arguments",
                               lambda: torch.LongTensor((6, 0), 1, 1, 0))
        self.assertRaisesRegex(TypeError,
                               r"tensor\(\) missing 1 required positional argument: \"data\"",
                               lambda: torch.tensor().new_zeros((5, 5), 0))

    @skipIfTorchDynamo("will be re-enabled after #90892")
    def test_from_buffer(self):
        a = bytearray([1, 2, 3, 4])
        self.assertEqual(torch.ByteStorage.from_buffer(a).tolist(), [1, 2, 3, 4])
        shorts = torch.ShortStorage.from_buffer(a, 'big')
        self.assertEqual(shorts.size(), 2)
        self.assertEqual(shorts.tolist(), [258, 772])
        ints = torch.IntStorage.from_buffer(a, 'little')
        self.assertEqual(ints.size(), 1)
        self.assertEqual(ints[0], 67305985)
        f = bytearray([0x40, 0x10, 0x00, 0x00])
        floats = torch.FloatStorage.from_buffer(f, 'big')
        self.assertEqual(floats.size(), 1)
        self.assertEqual(floats[0], 2.25)

        f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
        bools = torch.BoolStorage.from_buffer(f, 'big')
        self.assertEqual(bools.size(), 8)
        self.assertEqual(bools.tolist(), [False, True, True, True, True, True, True, True])
        self.assertEqual(bools.type(), 'torch.BoolStorage')
        self.assertTrue(isinstance(bools, torch.BoolStorage))

        f = bytearray(b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9')
        bools = torch.BoolStorage.from_buffer(f, 'big')
        self.assertEqual(bools.size(), 19)

        f = bytearray(b'\0x4A')
        bools = torch.BoolStorage.from_buffer(f, 'big')
        self.assertEqual(bools.size(), 4)
        self.assertEqual(bools.tolist(), [False, True, True, True])
        bytes = torch.ByteStorage.from_buffer(a)
        self.assertEqual(bytes.nbytes(), 4)
        self.assertEqual(bytes.tolist(), [1, 2, 3, 4])
        self.assertTrue(isinstance(bytes, torch.ByteStorage))

    def test_storage_error(self):
        quantized_storages = [
            torch.QInt32Storage,
            torch.QInt8Storage,
            torch.QUInt2x4Storage,
            torch.QUInt4x2Storage,
            torch.QUInt8Storage,
        ]

        with self.assertRaisesRegex(RuntimeError, r"Only child classes of _LegacyStorage can be instantiated"):
            torch.storage._LegacyStorage()

        for storage_class in torch._storage_classes:
            if storage_class in [torch.UntypedStorage, torch.TypedStorage]:
                continue

            device = 'cuda' if storage_class.__module__ == 'torch.cuda' else 'cpu'
            dtype = storage_class.dtype

            if device == 'cuda' and not torch.cuda.is_available():
                continue

            # Legacy <type>Storage constructor errors
            with self.assertRaisesRegex(RuntimeError, r"'device' cannot be specified"):
                storage_class(device='cpu')

            with self.assertRaisesRegex(RuntimeError, r"'dtype' cannot be specified"):
                storage_class(dtype=torch.float)

            with self.assertRaisesRegex(TypeError, r"got an unexpected keyword"):
                storage_class(sdlkjf=torch.float)

            with self.assertRaisesRegex(RuntimeError, r"Too many positional arguments"):
                storage_class(0, 0)

            with self.assertRaisesRegex(TypeError, r"invalid data type"):
                storage_class('string')

            with self.assertRaisesRegex(TypeError, r"Argument type not recognized"):
                storage_class(torch.tensor([]))

            s = storage_class()

            with self.assertRaisesRegex(RuntimeError, r"No positional arguments"):
                storage_class(0, wrap_storage=s.untyped())

            with self.assertRaisesRegex(TypeError, r"must be UntypedStorage"):
                storage_class(wrap_storage=s)

            if torch.cuda.is_available():
                if storage_class in quantized_storages:
                    with self.assertRaisesRegex(RuntimeError, r"Cannot create CUDA storage with quantized dtype"):
                        s.cuda()

                else:

                    if s.is_cuda:
                        s_other_device = s.cpu()
                    else:
                        s_other_device = s.cuda()

                    with self.assertRaisesRegex(RuntimeError, r"Device of 'wrap_storage' must be"):
                        storage_class(wrap_storage=s_other_device.untyped())

            # TypedStorage constructor errors
            with self.assertRaisesRegex(RuntimeError, r"No positional arguments"):
                torch.TypedStorage(0, wrap_storage=s.untyped(), dtype=dtype)

            with self.assertRaisesRegex(RuntimeError, r"Argument 'dtype' must be specified"):
                torch.TypedStorage(wrap_storage=s.untyped())

            with self.assertRaisesRegex(TypeError, r"Argument 'dtype' must be torch.dtype"):
                torch.TypedStorage(wrap_storage=s.untyped(), dtype=0)

            with self.assertRaisesRegex(RuntimeError, r"Argument 'device' should not be specified"):
                torch.TypedStorage(wrap_storage=s.untyped(), dtype=dtype, device=device)

            with self.assertRaisesRegex(TypeError, r"Argument 'wrap_storage' must be UntypedStorage"):
                torch.TypedStorage(wrap_storage=s, dtype=dtype)

            with self.assertRaisesRegex(RuntimeError, r"Storage device not recognized"):
                torch.TypedStorage(dtype=dtype, device='xla')

            if torch.cuda.is_available():
                if storage_class in quantized_storages:
                    with self.assertRaisesRegex(RuntimeError, r"Cannot create CUDA storage with quantized dtype"):
                        torch.TypedStorage(dtype=dtype, device='cuda')

            with self.assertRaisesRegex(TypeError, r"Argument type not recognized"):
                torch.TypedStorage(torch.tensor([]), dtype=dtype, device=device)

            with self.assertRaisesRegex(RuntimeError, r"Too many positional arguments"):
                torch.TypedStorage(0, 0, dtype=dtype, device=device)

            if isinstance(s, torch.TypedStorage):
                s_other = torch.TypedStorage([1, 2, 3, 4], device=device, dtype=dtype)

                with self.assertRaisesRegex(RuntimeError, r'cannot set item'):
                    s.fill_(s_other)

    def test_storage_error_no_attribute(self):
        storage_classes = [
            torch.cuda.ByteStorage,
            torch.cuda.FloatStorage,
        ]
        for storage_class in storage_classes:
            with self.assertRaisesRegex(RuntimeError, r'Not available for CUDA storage'):
                storage_class.from_buffer()

            with self.assertRaisesRegex(RuntimeError, r'Not available for CUDA storage'):
                storage_class._new_with_weak_ptr()

            with self.assertRaisesRegex(RuntimeError, r'Not available for CUDA storage'):
                storage_class._new_shared_filename(0, 0, 0)

    def test_storage_casts(self):
        storage = torch.IntStorage([-1, 0, 1, 2, 3, 4])
        self.assertEqual(storage.size(), 6)
        self.assertEqual(storage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(storage.type(), 'torch.IntStorage')
        self.assertIs(storage.dtype, torch.int32)

        floatStorage = storage.float()
        self.assertEqual(floatStorage.size(), 6)
        self.assertEqual(floatStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(floatStorage.type(), 'torch.FloatStorage')
        self.assertEqual(floatStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(floatStorage.dtype, torch.float32)

        halfStorage = storage.half()
        self.assertEqual(halfStorage.size(), 6)
        self.assertEqual(halfStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(halfStorage.type(), 'torch.HalfStorage')
        self.assertEqual(halfStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(halfStorage.dtype, torch.float16)

        bfloat16Storage = storage.bfloat16()
        self.assertEqual(bfloat16Storage.size(), 6)
        self.assertEqual(bfloat16Storage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(bfloat16Storage.type(), 'torch.BFloat16Storage')
        self.assertEqual(bfloat16Storage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(bfloat16Storage.dtype, torch.bfloat16)

        longStorage = storage.long()
        self.assertEqual(longStorage.size(), 6)
        self.assertEqual(longStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(longStorage.type(), 'torch.LongStorage')
        self.assertEqual(longStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(longStorage.dtype, torch.int64)

        shortStorage = storage.short()
        self.assertEqual(shortStorage.size(), 6)
        self.assertEqual(shortStorage.tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertEqual(shortStorage.type(), 'torch.ShortStorage')
        self.assertEqual(shortStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(shortStorage.dtype, torch.int16)

        doubleStorage = storage.double()
        self.assertEqual(doubleStorage.size(), 6)
        self.assertEqual(doubleStorage.tolist(), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(doubleStorage.type(), 'torch.DoubleStorage')
        self.assertEqual(doubleStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(doubleStorage.dtype, torch.float64)

        charStorage = storage.char()
        self.assertEqual(charStorage.size(), 6)
        self.assertEqual(charStorage.tolist(), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(charStorage.type(), 'torch.CharStorage')
        self.assertEqual(charStorage.int().tolist(), [-1, 0, 1, 2, 3, 4])
        self.assertIs(charStorage.dtype, torch.int8)

        byteStorage = storage.byte()
        self.assertEqual(byteStorage.size(), 6)
        self.assertEqual(byteStorage.tolist(), [255, 0, 1, 2, 3, 4])
        self.assertEqual(byteStorage.type(), 'torch.ByteStorage')
        self.assertEqual(byteStorage.int().tolist(), [255, 0, 1, 2, 3, 4])
        self.assertIs(byteStorage.dtype, torch.uint8)

        boolStorage = storage.bool()
        self.assertEqual(boolStorage.size(), 6)
        self.assertEqual(boolStorage.tolist(), [True, False, True, True, True, True])
        self.assertEqual(boolStorage.type(), 'torch.BoolStorage')
        self.assertEqual(boolStorage.int().tolist(), [1, 0, 1, 1, 1, 1])
        self.assertIs(boolStorage.dtype, torch.bool)

        complexfloat_storage = torch.ComplexFloatStorage([-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j])
        self.assertEqual(complexfloat_storage.size(), 6)
        self.assertEqual(complexfloat_storage.tolist(), [-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j])
        self.assertEqual(complexfloat_storage.type(), 'torch.ComplexFloatStorage')
        self.assertIs(complexfloat_storage.dtype, torch.complex64)

        complexdouble_storage = complexfloat_storage.complex_double()
        self.assertEqual(complexdouble_storage.size(), 6)
        self.assertEqual(complexdouble_storage.tolist(), [-1, 0, 1 + 2j, 2.5j, 3.5, 4 - 2j])
        self.assertEqual(complexdouble_storage.type(), 'torch.ComplexDoubleStorage')
        self.assertIs(complexdouble_storage.dtype, torch.complex128)

    # Test that internal versions of functions related to TypedStorage do not
    # produce a deprecation warning
    def test_typed_storage_internal_no_warning(self):
        s0 = torch.FloatStorage(10)
        s0_untyped = s0.untyped()
        t0 = torch.randn(10)

        funcs = [
            lambda: torch.FloatStorage(_internal=True),
            lambda: torch.TypedStorage(
                dtype=torch.float,
                device='cpu',
                _internal=True),
            lambda: torch.TypedStorage(
                wrap_storage=s0_untyped,
                dtype=s0.dtype,
                _internal=True),
            lambda: torch.FloatStorage._dtype,
            lambda: s0._resize_(20),
            lambda: s0._size(),
            lambda: s0._untyped_storage,
            lambda: s0._is_shared(),
            lambda: s0._share_memory_(),
            lambda: s0._pickle_storage_type(),
            lambda: s0._setitem(slice(0, s0._size()), 1),
            lambda: s0._element_size(),
            lambda: s0._deepcopy({}),
            lambda: s0._data_ptr(),
            lambda: s0._nbytes(),
            lambda: t0._typed_storage(),
        ]

        if torch.cuda.is_available():
            s1 = torch.cuda.FloatStorage(10)
            s1_untyped = s1.untyped()
            t1 = torch.randn(10, device='cuda')

            funcs += [
                lambda: torch.cuda.FloatStorage(_internal=True),
                lambda: torch.TypedStorage(
                    dtype=torch.float,
                    device='cuda',
                    _internal=True),
                lambda: torch.TypedStorage(
                    wrap_storage=s1_untyped,
                    dtype=s1.dtype,
                    _internal=True),
                lambda: torch.cuda.FloatStorage._dtype,
                lambda: s1._resize_(20),
                lambda: s1._size(),
                lambda: s1._untyped_storage,
                lambda: s1._is_shared(),
                lambda: s1._share_memory_(),
                lambda: s1._pickle_storage_type(),
                lambda: s1._setitem(slice(0, s1._size()), 1),
                lambda: s1._element_size(),
                lambda: s1._deepcopy({}),
                lambda: s1._data_ptr(),
                lambda: s1._nbytes(),
                lambda: t1._typed_storage(),
            ]

        # Check that each of the TypedStorage internal function calls do not
        # produce a deprecation warning
        for f in funcs:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', "TypedStorage is deprecated")
                f()

    # Test that public functions related to TypedStorage produce a deprecation
    # warning
    @skipIfTorchInductor("FIXME")
    def test_typed_storage_deprecation_warning(self):
        s0 = torch.FloatStorage(10)
        funcs = [
            lambda: torch.FloatStorage(),
            lambda: torch.FloatStorage.dtype,
            lambda: s0.fill_(0),
            lambda: s0.is_cuda,
            lambda: s0.untyped(),
            lambda: len(s0),
            lambda: s0[0],
        ]

        if torch.cuda.is_available():
            s1 = torch.cuda.FloatStorage(10)
            funcs += [
                lambda: torch.cuda.FloatStorage(),
                lambda: torch.cuda.FloatStorage.dtype,
                lambda: s1.fill_(0),
                lambda: s1.is_cuda,
                lambda: s1.untyped(),
                lambda: len(s1),
                lambda: s1[0],
            ]

        # Check that each of the TypedStorage function calls produce a warning
        # if warnings are reset between each
        for f in funcs:
            with warnings.catch_warnings(record=True) as w:
                warnings.resetwarnings()
                f()
                self.assertEqual(len(w), 1, msg=str([str(a) for a in w]))
                warning = w[0].message
                self.assertTrue(warning, DeprecationWarning)
                self.assertTrue(re.search(
                    '^TypedStorage is deprecated',
                    str(warning)))

    def test_from_file(self):
        def assert_with_filename(filename):
            size = 10000
            s1 = torch.FloatStorage.from_file(filename, True, size)
            t1 = torch.FloatTensor(s1).copy_(torch.randn(size))
            self.assertEqual(s1.data_ptr(), torch.FloatTensor(s1).data_ptr())

            # check mapping
            s2 = torch.FloatStorage.from_file(filename, True, size)
            t2 = torch.FloatTensor(s2)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # check changes to t1 from t2
            rnum = random.uniform(-1, 1)
            t1.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # check changes to t2 from t1
            rnum = random.uniform(-1, 1)
            t2.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # release the tensors
            del s1, t1, s2, t2

        with TemporaryFileName() as fname:
            assert_with_filename(fname)

        if IS_FILESYSTEM_UTF8_ENCODING:
            with TemporaryDirectoryName(suffix='中文') as dname, TemporaryFileName(dir=dname) as fname:
                assert_with_filename(fname)

    def test_torch_from_file(self):
        def assert_with_filename(filename):
            size = 10000
            s1 = torch.from_file(filename, True, size, dtype=torch.float)
            t1 = torch.FloatTensor(s1).copy_(torch.randn(size))

            # check mapping
            s2 = torch.from_file(filename, True, size, dtype=torch.float)
            t2 = torch.FloatTensor(s2)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # check changes to t1 from t2
            rnum = random.uniform(-1, 1)
            t1.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # check changes to t2 from t1
            rnum = random.uniform(-1, 1)
            t2.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # release the tensors
            del s1, t1, s2, t2

        with TemporaryFileName() as fname:
            assert_with_filename(fname)

        if IS_FILESYSTEM_UTF8_ENCODING:
            with TemporaryDirectoryName(suffix='中文') as dname, TemporaryFileName(dir=dname) as fname:
                assert_with_filename(fname)

    def test_print(self):
        default_type = torch.tensor([]).type()
        for t in torch._tensor_classes:
            if t == torch.HalfTensor:
                continue  # HalfTensor does not support fill
            if t.is_sparse:
                continue
            if t.is_cuda and not torch.cuda.is_available():
                continue
            obj = t(100, 100).fill_(1)
            obj.__repr__()
            str(obj)
        # test half tensor
        obj = torch.rand(100, 100, device='cpu').half()
        obj.__repr__()
        str(obj)
        for t in torch._storage_classes:
            if t == torch.BFloat16Storage:
                continue  # Fix once fill is enabled for bfloat16
            if t.is_cuda and not torch.cuda.is_available():
                continue
            if t == torch.BoolStorage or t == torch.cuda.BoolStorage:
                obj = t(100).fill_(True)
            else:
                obj = t(100).fill_(1)
            obj.__repr__()
            str(obj)

        # test complex tensor
        # complex tensor print uses two formatters, one for real values
        # and the other for imag values. this is consistent with numpy
        x = torch.tensor([2.3 + 4j, 7 + 6j])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([2.3000+4.j, 7.0000+6.j])''')

        # test complex half tensor
        x = torch.tensor([1.25 + 4j, -7. + 6j], dtype=torch.chalf)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([ 1.2500+4.j, -7.0000+6.j], dtype=torch.complex32)''')

        # test scientific notation for complex tensors
        x = torch.tensor([1e28 + 2j , -1e-28j])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e+28+2.0000e+00j, -0.0000e+00-1.0000e-28j])''')

        # test big integer
        x = torch.tensor(2341234123412341)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor(2341234123412341)''')

        # test scientific notation
        x = torch.tensor([1e28, 1e-28])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e+28, 1.0000e-28])''')

        # test scientific notation using set_printoptions
        x = torch.tensor([1e2, 1e-2])
        torch.set_printoptions(sci_mode=True)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e+02, 1.0000e-02])''')
        torch.set_printoptions(sci_mode=False)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([  100.0000,     0.0100])''')
        torch.set_printoptions(sci_mode=None)  # reset to the default value

        # test no leading space if all elements positive
        x = torch.tensor([1, 2])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1, 2])''')

        # test for leading space if there are negative elements
        x = torch.tensor([1, -2])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([ 1, -2])''')

        # test inf and nan
        x = torch.tensor([4, inf, 1.5, -inf, 0, nan, 1])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([4.0000,    inf, 1.5000,   -inf, 0.0000,    nan, 1.0000])''')

        y = torch.tensor([4, inf, complex(1.5, inf), complex(-inf, 4), 0, complex(nan, inf), complex(3, nan)])
        self.assertEqual(y.__repr__(), str(y))
        expected_str = '''\
tensor([4.0000+0.j,    inf+0.j, 1.5000+infj,   -inf+4.j, 0.0000+0.j,    nan+infj,
        3.0000+nanj])'''
        self.assertExpectedInline(str(y), expected_str)

        # test dtype
        torch.set_default_dtype(torch.float)
        x = torch.tensor([1e-324, 1e-323, 1e-322, 1e307, 1e308, 1e309], dtype=torch.float64)
        self.assertEqual(x.__repr__(), str(x))
        expected_str = '''\
tensor([ 0.0000e+00, 9.8813e-324, 9.8813e-323, 1.0000e+307, 1.0000e+308,
                inf], dtype=torch.float64)'''
        self.assertExpectedInline(str(x), expected_str)

        # test changing default dtype
        torch.set_default_dtype(torch.float64)
        self.assertEqual(x.__repr__(), str(x))
        expected_str = '''\
tensor([ 0.0000e+00, 9.8813e-324, 9.8813e-323, 1.0000e+307, 1.0000e+308,
                inf])'''
        self.assertExpectedInline(str(x), expected_str)

        # test summary
        x = torch.zeros(10000)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([0., 0., 0.,  ..., 0., 0., 0.])''')

        # test internal summary function
        x = torch.rand(1, 20, 5, 30)
        summary = torch._tensor_str.get_summarized_data(x)
        self.assertEqual(summary.shape, (1, 6, 5, 6))
        first_and_last = [0, 1, 2, -3, -2, -1]
        self.assertEqual(summary, x[:, first_and_last][..., first_and_last])

        # test device
        if torch.cuda.is_available():
            x = torch.tensor([123], device='cuda:0')
            self.assertEqual(x.__repr__(), str(x))
            self.assertExpectedInline(str(x), '''tensor([123], device='cuda:0')''')

            # test changing default to cuda
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.assertEqual(x.__repr__(), str(x))
            self.assertExpectedInline(str(x), '''tensor([123])''')

            # test printing a tensor on a different gpu than current one.
            if torch.cuda.device_count() >= 2:
                with torch.cuda.device(1):
                    self.assertEqual(x.__repr__(), str(x))
                    self.assertExpectedInline(str(x), '''tensor([123], device='cuda:0')''')

            # test printing cpu tensor when default device is cuda
            y = torch.tensor([123], device='cpu')
            self.assertEqual(y.__repr__(), str(y))
            self.assertExpectedInline(str(y), '''tensor([123], device='cpu')''')
        torch.set_default_tensor_type(default_type)


        # test integral floats and requires_grad
        x = torch.tensor([123.], requires_grad=True)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([123.], requires_grad=True)''')

        # test non-contiguous print
        # sliced tensor should have > PRINT_OPTS.threshold elements
        x = torch.ones(100, 2, 2, 10)
        y = x.as_strided(size=(100, 2, 10), stride=(2 * 2 * 10, 2 * 10, 1))
        self.assertEqual(str(y), y.__repr__())
        expected_str = '''\
tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        ...,

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]]])\
'''

        self.assertExpectedInline(str(y), expected_str)

        x = torch.ones(100, 2, 2, 10) * (1 + 1j)
        y = x.as_strided(size=(100, 2, 10), stride=(2 * 2 * 10, 2 * 10, 1))
        self.assertEqual(str(y), y.__repr__())
        expected_str = '''\
tensor([[[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        ...,

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]]])\
'''
        self.assertExpectedInline(str(y), expected_str)

        # test print 0-dim tensor: there's no 0-dim in Numpy, we match arrayprint style
        x = torch.tensor(0.00002)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor(2.0000e-05)''')

        # test print boolean tensor
        x = torch.tensor([True])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([True])''')

        x = torch.tensor(True)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor(True)''')

        # [Numpy] test print float in sci_mode when min < 0.0001.
        x = torch.tensor([0.00002])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([2.0000e-05])''')

        # [Numpy] test print complex in sci_mode when real_min < 0.0001 and (or) imag_min < 0.0001.
        x = torch.tensor([0.00002]) * (1 + 1j)
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([2.0000e-05+2.0000e-05j])''')

        # [Numpy] test print float in sci_mode when max > 1e8.
        # TODO: Pytorch uses fixed precision to print, while Numpy uses dragon4_scientific
        # to do automatic trimming and padding.
        x = torch.tensor([123456789.])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.2346e+08])''')

        # [Numpy] test print float in sci_mode when max / min > 1000.
        x = torch.tensor([0.01, 11])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e-02, 1.1000e+01])''')

        # [Numpy] test print int max / min > 1000, no sci_mode
        x = torch.tensor([1, 1010])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([   1, 1010])''')

        # [Numpy] test print int > 1e8, no sci_mode
        x = torch.tensor([1000000000])  # 1e9
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1000000000])''')

        # [Numpy] test printing float in int_mode
        x = torch.tensor([1., 1000.])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([   1., 1000.])''')

        # [Numpy] test printing float in int_mode in sci format when max / min > 1000.
        x = torch.tensor([1., 1010.])
        self.assertEqual(x.__repr__(), str(x))
        self.assertExpectedInline(str(x), '''tensor([1.0000e+00, 1.0100e+03])''')

    def test_sizeof(self) -> None:
        sizeof_empty = torch.randn(0).storage().__sizeof__()
        sizeof_10 = torch.randn(10).storage().__sizeof__()
        sizeof_100 = torch.randn(100).storage().__sizeof__()
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

        sizeof_empty = torch.randn(0).to(torch.uint8).storage().__sizeof__()
        sizeof_10 = torch.randn(10).to(torch.uint8).storage().__sizeof__()
        sizeof_100 = torch.randn(100).to(torch.uint8).storage().__sizeof__()
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

    def test_iter(self) -> None:
        x = torch.randn(5, 5)
        for i, sub in enumerate(x):
            self.assertEqual(sub, x[i])

        x = torch.tensor([])
        self.assertEqual(list(x), [])

    def test_new(self) -> None:
        x = torch.autograd.Variable(torch.tensor([]))
        y = torch.autograd.Variable(torch.randn(4, 4))
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))
        self.assertEqual(x.new().shape, [0])
        self.assertEqual(x.new(), x)
        self.assertEqual(x.new(1, 2).shape, [1, 2])
        self.assertEqual(x.new(torch.Size([3, 4])).shape, [3, 4])
        self.assertEqual(x.new([3, 4]).shape, [2])
        self.assertEqual(x.new([3, 4]).tolist(), [3, 4])
        self.assertEqual(x.new((3, 4)).tolist(), [3, 4])
        self.assertEqual(x.new([np.int32(3), np.float64(4)]).tolist(), [3, 4])
        self.assertEqual(x.new(np.array((3, 4))).tolist(), [3, 4])
        self.assertEqual(x.new([z[2], z[0] + 3]).tolist(), [3, 4])
        self.assertEqual(x.new(size=(3, 4)).shape, [3, 4])
        self.assertEqual(x.new(()).shape, [0])
        self.assertEqual(x.new(y.storage()).data_ptr(), y.data_ptr())
        self.assertEqual(x.new(y).data_ptr(), y.data_ptr())
        self.assertIsNot(x.new(y), y)

        self.assertRaises(TypeError, lambda: x.new(z))
        # TypeError would be better
        self.assertRaises(RuntimeError, lambda: x.new(z.storage()))

    @unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    @skipIfTorchInductor("pin_memory isn't yet supported in TorchInductor")
    def test_pin_memory(self):
        x = torch.randn(3, 5)
        self.assertFalse(x.is_pinned())
        if not torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: x.pin_memory())
        else:
            pinned = x.pin_memory()
            self.assertTrue(pinned.is_pinned())
            self.assertEqual(pinned, x)
            self.assertNotEqual(pinned.data_ptr(), x.data_ptr())
            # test that pin_memory on already pinned tensor has no effect
            self.assertIs(pinned, pinned.pin_memory())
            self.assertEqual(pinned.data_ptr(), pinned.pin_memory().data_ptr())

    def test_error_msg_type_translation(self):
        with self.assertRaisesRegex(
                RuntimeError,
                # message includes both Double and Long
                '(?=.*Double)(?=.*Long)'):

            # Calls model with a LongTensor input but DoubleTensor weights
            input = torch.zeros(1, 1, 1, 6, dtype=torch.long)
            weight = torch.nn.Parameter(torch.zeros(1, 1, 1, 3, dtype=torch.double))
            model = torch.nn.Conv2d(1, 1, (1, 3), stride=1, padding=0, bias=False)
            model.weight = weight
            out = model(input)

    def test_apply(self):
        x = torch.arange(1, 6)
        res = x.clone().apply_(lambda k: k + k)
        self.assertEqual(res, x * 2)
        self.assertRaises(TypeError, lambda: x.apply_(lambda k: "str"))

    def test_map(self):
        x = torch.autograd.Variable(torch.randn(3, 3))
        y = torch.autograd.Variable(torch.randn(3))
        res = x.clone()
        res.map_(y, lambda a, b: a + b)
        self.assertEqual(res, x + y)
        self.assertRaisesRegex(TypeError, "not callable", lambda: res.map_(y, "str"))

    def test_map2(self):
        x = torch.autograd.Variable(torch.randn(3, 3))
        y = torch.autograd.Variable(torch.randn(3))
        z = torch.autograd.Variable(torch.randn(1, 3))
        res = x.clone()
        res.map2_(y, z, lambda a, b, c: a + b * c)
        self.assertEqual(res, x + y * z)
        z.requires_grad = True
        self.assertRaisesRegex(
            RuntimeError, "requires grad",
            lambda: res.map2_(y, z, lambda a, b, c: a + b * c))

    def test_Size(self):
        x = torch.Size([1, 2, 3])
        self.assertIsInstance(x, tuple)
        self.assertEqual(x[0], 1)
        self.assertEqual(x[1], 2)
        self.assertEqual(x[2], 3)
        self.assertEqual(len(x), 3)
        self.assertRaises(TypeError, lambda: torch.Size(torch.ones(3)))

        self.assertIsInstance(x * 2, torch.Size)
        self.assertIsInstance(x[:-1], torch.Size)
        self.assertIsInstance(x + x, torch.Size)

    def test_Size_scalar(self):
        three = torch.tensor(3)
        two = torch.tensor(2)
        x = torch.Size([0, 1, two, three, 4])
        for i in range(1, 5):
            self.assertEqual(x[i], i)

    def test_Size_iter(self):
        for sizes in [iter([1, 2, 3, 4, 5]), range(1, 6)]:
            x = torch.Size(sizes)
            for i in range(0, 5):
                self.assertEqual(x[i], i + 1)

    def test_t_not_2d_error(self):
        self.assertRaises(RuntimeError, lambda: torch.randn(2, 3, 4).t())
        self.assertRaises(RuntimeError, lambda: torch.randn(2, 3, 4).t_())

    # skip this test for now as it affects all tests
    @unittest.skipIf(True, "flush_denormal not supported")
    def test_set_flush_denormal(self):
        tiny_float = 1e-42
        tiny_double = 1e-320
        float_tensor = torch.FloatTensor([1.0, tiny_float])
        double_tensor = torch.DoubleTensor([1.0, tiny_float, tiny_double])

        self.assertEqual(float_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(float_tensor[1], tiny_float, atol=tiny_float / 16, rtol=0)
        self.assertEqual(double_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[1], tiny_float, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[2], tiny_double, atol=0.0, rtol=0)

        torch.set_flush_denormal(True)
        self.assertEqual(float_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(float_tensor[1], 0.0, atol=0.0, rtol=0)  # tiny_float to zero
        self.assertEqual(double_tensor[0], 1.0, atol=0.0, rtol=0)
        # tiny_float is not converted to zero in double type
        self.assertEqual(double_tensor[1], tiny_float, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[2], 0.0, atol=0.0, rtol=0)  # tiny_double to zero
        torch.set_flush_denormal(False)

    def test_show_config(self):
        # We can't usefully test the output; just make sure this doesn't crash
        torch.__config__.show()

    @unittest.skipIf(IS_FBCODE, "CXX_FLAGS is only for OSS build.")
    def test_cxx_flags(self):
        torch.__config__._cxx_flags()

    def test_parallel_info(self):
        torch.__config__.parallel_info()

    @slowTest
    def test_slow_test(self):
        # Just a smoketest to make sure our slowTest decorator works.
        pass

    def test_is_nonzero(self):
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch.tensor([]).is_nonzero()
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch.tensor([0, 0]).is_nonzero()
        self.assertFalse(torch.tensor(0).is_nonzero())
        self.assertTrue(torch.tensor(1).is_nonzero())
        self.assertFalse(torch.tensor([0]).is_nonzero())
        self.assertTrue(torch.tensor([1]).is_nonzero())
        self.assertFalse(torch.tensor([[0]]).is_nonzero())
        self.assertTrue(torch.tensor([[1]]).is_nonzero())
        self.assertTrue(torch.tensor(0.1).is_nonzero())
        self.assertTrue(torch.tensor(-0.1).is_nonzero())
        self.assertFalse(torch.tensor(0.0).is_nonzero())
        self.assertTrue(torch.tensor(True).is_nonzero())
        self.assertFalse(torch.tensor(False).is_nonzero())
        self.assertFalse(torch.tensor(0 + 0j).is_nonzero())
        self.assertTrue(torch.tensor(0 + 0.1j).is_nonzero())

    def test_assert_async(self):
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch._assert_async(torch.tensor([]))
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch._assert_async(torch.tensor([0, 0]))
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0))
        torch._assert_async(torch.tensor(1))
        torch._assert_async(torch.tensor(0.1))
        torch._assert_async(torch.tensor(-0.1))
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0.0))
        torch._assert_async(torch.tensor(True))
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(False))
        torch._assert_async(torch.tensor(0 + 0.1j))
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0 + 0j))

    # NB: we must not be built with CUDA; if we are built with CUDA but no CUDA
    # is available, we get a different error.
    @unittest.skipIf(torch.backends.cuda.is_built() or IS_SANDCASTLE, "CUDA is built, can't test CUDA not built error")
    def test_cuda_not_built(self):
        msg = "Torch not compiled with CUDA enabled"
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.cuda.current_device())
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1], device="cuda"))
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1]).cuda())
        self.assertRaisesRegex(TypeError, msg, lambda: torch.cuda.FloatTensor())
        self.assertRaisesRegex(TypeError, msg, lambda: torch.set_default_tensor_type(torch.cuda.FloatTensor))
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1]).to(device="cuda"))

    def test_has_internal_overlap(self):
        OVERLAP_NO = 0
        OVERLAP_YES = 1
        OVERLAP_TOO_HARD = 2

        # Check for contiguous tensors
        a = torch.randn(3, 3)
        self.assertEqual(torch._debug_has_internal_overlap(a), OVERLAP_NO)

        # Checks for zero strides
        b = torch.randn(1, 3)
        b_expanded = b.expand(4, 3)
        self.assertEqual(torch._debug_has_internal_overlap(b_expanded), OVERLAP_YES)

        # Check for zero strided, size 1 axis, in non-contiguous storage (gh-33812)
        c = torch.randn(10).as_strided([2, 1, 5], [1, 0, 2])
        self.assertEqual(torch._debug_has_internal_overlap(c), OVERLAP_NO)
        c = torch.randn(2, 1, 10)[::2].as_strided((2, 1, 5), (10, 0, 2))
        self.assertEqual(torch._debug_has_internal_overlap(c), OVERLAP_TOO_HARD)

    def test_allow_tensor_metadata_change(self):
        a = torch.ones(2, 3)
        # Metadata changes are allowed on view tensors that are created from detach().

    @skipIfNotRegistered("LayerNorm", "Skipping as LayerNorm is not registered")
    def test_c10_layer_norm(self):
        # test that we can call c10 ops and they return a reasonable result
        X = torch.rand(5, 5, dtype=torch.float)
        weight = torch.rand(*X.size()[1:], dtype=torch.float)
        bias = torch.rand(*X.size()[1:], dtype=torch.float)
        epsilon = 1e-4

        expected_norm = torch.nn.functional.layer_norm(
            X, X.size()[1:], weight=weight, bias=bias, eps=epsilon)
        actual_norm, actual_mean, actual_stdev = \
            torch.ops._caffe2.LayerNorm(torch.tensor(X), torch.tensor(
                weight), torch.tensor(bias), 1, epsilon, True)
        torch.testing.assert_close(expected_norm, actual_norm)

    @skipIfTorchInductor("To be supported")
    def test_memory_format(self):
        def test_helper(x, memory_format):
            y = x.contiguous(memory_format=memory_format)
            self.assertFalse(y.is_contiguous())
            self.assertTrue(y.is_contiguous(memory_format=memory_format))
            self.assertEqual(y, x)

        test_helper(torch.randn(4, 3, 8, 8), torch.channels_last)
        test_helper(torch.randn(4, 3, 8, 8, 8), torch.channels_last_3d)

    def test_memory_format_contiguous_returns_same_tensor_if_already_satisfies(self):
        def test_helper(x, memory_format):
            alias = x.contiguous(memory_format=memory_format)
            alias.fill_(7)
            self.assertEqual(x, alias)

        test_helper(torch.randn(4, 8, 8, 3).permute(0, 3, 1, 2), torch.channels_last)
        test_helper(torch.randn(4, 8, 8, 8, 3).permute(0, 4, 1, 2, 3), torch.channels_last_3d)

    def test_memory_format_empty(self):
        def test_helper(dim1, dim2, memory_format):
            with self.assertRaises(RuntimeError):
                x = torch.empty(dim1, memory_format=memory_format)
            x = torch.empty(dim2, memory_format=memory_format)
            self.assertTrue(x.is_contiguous(memory_format=memory_format))

        test_helper((3, 3), (3, 3, 3, 3), torch.channels_last)
        test_helper((3, 3, 3), (3, 3, 3, 3, 3), torch.channels_last_3d)

    def test_subclass_tensors(self):
        # raise an error when trying to subclass FloatTensor
        with self.assertRaisesRegex(TypeError, "type 'torch.FloatTensor' is not an acceptable base type"):
            class Foo1(torch.FloatTensor):
                pass

        # but allow subclassing Tensor:
        class Foo2(torch.Tensor):
            def foo(self):
                return 5
        f = Foo2()
        self.assertEqual(f.foo(), 5)

    def test_ndim(self):
        a = torch.randn(1, 2, 3)
        self.assertEqual(3, a.ndim)
        b = torch.randn(())
        self.assertEqual(0, b.ndim)
        c = torch.randn(1, 0)
        self.assertEqual(2, c.ndim)

    def test_fill_diagonal(self):
        a1 = torch.randn(7, 3)
        a2 = a1.clone()
        v = 1
        for i in range(3):
            a2[i][i] = v
        a1.fill_diagonal_(v)
        self.assertEqual(a1, a2)

        b1 = torch.randn(7, 3)
        b2 = b1.clone()
        for i in range(3):
            b2[i][i] = v
            b2[i + 4][i] = v
        b1.fill_diagonal_(v, wrap=True)
        self.assertEqual(b1, b2)

        c1 = torch.rand(3, 3, 3)
        c2 = c1.clone()
        for i in range(3):
            c2[i][i][i] = v
        c1.fill_diagonal_(v)
        self.assertEqual(c1, c2)

        # non-contiguous tensor
        d1 = torch.rand(3, 3, 3)[:, 1, ...]
        d2 = d1.clone()
        for i in range(3):
            d2[i][i] = v
        d1.fill_diagonal_(v)
        self.assertEqual(d1, d2)

        e1 = torch.rand(7, 3, 3)[:, 1, ...]
        e2 = e1.clone()
        for i in range(3):
            e2[i][i] = v
            e2[i + 4][i] = v
        e1.fill_diagonal_(v, wrap=True)
        self.assertEqual(e1, e2)

    def test_setting_real_imag_to_a_number(self):
        x = torch.randn(4, dtype=torch.cfloat)
        x.real = 0
        x.imag = 0
        zeros = torch.zeros(4)
        self.assertEqual(x.real, zeros)
        self.assertEqual(x.imag, zeros)

    def test_batch_norm_cpu_inference(self):
        # input nchw in (2,1,1,1), (2,2,2,2)
        inputs = [
            torch.tensor([[[[-0.5000]]], [[[0.5000]]]]),
            torch.tensor([
                [
                    [[-0.5000, 0.5000], [-1.0000, 1.0000]],
                    [[-0.2500, -0.5000], [0.2500, 0.5000]]
                ],
                [
                    [[0.1000, 1.0000], [1.0000, 0.1000]],
                    [[1.0000, 0.5000], [1.5000, -1.5000]]
                ]])]
        # output nchw in (2,1,1,1), (2,2,2,2)
        outputs = [
            torch.tensor([
                [[[-0.499997496604919433593750000]]],
                [[[0.499997496604919433593750000]]]]),
            torch.tensor([
                [[[-0.499997496604919433593750000, 0.499997496604919433593750000],
                  [-0.999994993209838867187500000, 0.999994993209838867187500000]],
                 [[-0.249998748302459716796875000, -0.499997496604919433593750000],
                  [0.249998748302459716796875000, 0.499997496604919433593750000]]],
                [[[0.099999502301216125488281250, 0.999994993209838867187500000],
                  [0.999994993209838867187500000, 0.099999502301216125488281250]],
                 [[0.999994993209838867187500000, 0.499997496604919433593750000],
                  [1.499992489814758300781250000, -1.499992489814758300781250000]]]])]


        for i in range(len(inputs)):
            for affine in [False, True]:
                m = torch.nn.BatchNorm2d(inputs[i].size()[1], 1e-05, 0.1, affine=affine)
                m.eval()
                # contiguous case
                input1 = inputs[i].contiguous()
                output1 = m(input1)
                # non-contiguous case
                input2 = input1.permute(0, 1, 3, 2)
                output2 = m(input2).permute(0, 1, 3, 2)
                # channels last case
                input3 = input1.contiguous(memory_format=torch.channels_last)
                output3 = m(input3)
                self.assertEqual(output3, outputs[i])
                self.assertEqual(output3, output1)
                self.assertEqual(output3, output2)

    # FIXME: move these meta tests to their own test suite/class or
    #   distribute them among the appropriate test suites for their ops
    def test_empty_meta(self):
        x = torch.empty(2 ** 20, 2 ** 20, device='meta')
        y = torch.empty(2 ** 20, device='meta')
        z = x + y
        self.assertEqual(z.size(), (2 ** 20, 2 ** 20))
        self.assertRaises(RuntimeError, lambda: z[0][0].item())

    def test_format_scalar_meta(self):
        x = torch.empty((), device='meta')
        self.assertEqual(format(x), repr(x))

    def test_upsample_nearest1d_meta(self):
        # TODO: this test should be triggered by test_nn.py but right
        # now meta is not enabled (and even if it was, we are probably
        # missing too many meta functions to get through the test unmolested)

        # NB: Can't make the exponent too big, or it will overflow
        # signed 64-bit integer
        x = torch.empty(2 * 10 ** 8, 3, 2 * 10 ** 8, device='meta')
        z = torch.nn.functional.interpolate(x, scale_factor=2)
        self.assertEqual(z.size(), (2 * 10 ** 8, 3, 4 * 10 ** 8))
        self.assertRaises(RuntimeError, lambda: z[0][0][0].item())

        # TODO: the out tests cannot be triggered by test_nn.py because
        # we don't actually do out= arguments for nn functions, so there
        # is no public API by which to get the out version

        # interpolate doesn't seem to support out=
        # (not sure why passing None here doesn't work? How strange...)
        z = torch.empty(0, device='meta')
        torch._C._nn.upsample_nearest1d(x, (4 * 10 ** 8,), 2, out=z)
        self.assertEqual(z.size(), (2 * 10 ** 8, 3, 4 * 10 ** 8))
        self.assertRaises(RuntimeError, lambda: z[0][0][0].item())

    def test_upsample_nearest2d_meta(self):
        # TODO: the out tests cannot be triggered by test_nn.py because
        # we don't actually do out= arguments for nn functions, so there
        # is no public API by which to get the out version

        # Make sure we don't clobber strides of out tensor.  NB: this
        # test must be done on 2d/3d, because 1d doesn't have any meaningful
        # layout support
        x = torch.empty(4, 3, 8, 8, device='meta')
        out = torch.empty(4, 3, 16, 16, device='meta', memory_format=torch.channels_last)
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        x = torch.empty(4, 3, 8, 8, device='meta', memory_format=torch.channels_last)
        out = torch.empty(4, 3, 16, 16, device='meta')
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous())

        # But if resize occurs, do clobber
        x = torch.empty(4, 3, 8, 8, device='meta', memory_format=torch.channels_last)
        out = torch.empty(0, device='meta')
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        # Complain if out dtype mismatch
        x = torch.empty(4, 3, 8, 8, device='meta', dtype=torch.float)
        out = torch.empty(4, 3, 16, 16, device='meta', dtype=torch.double)
        self.assertExpectedRaisesInline(
            RuntimeError, lambda: torch._C._nn.upsample_nearest2d(x, (16, 16), out=out),
            """Expected out tensor to have dtype float, but got double instead"""
        )

        # Complain if out device mismatch
        x = torch.empty(0, 3, 8, 8, device='meta')
        out = torch.empty(0, 3, 16, 16, device='cpu')
        self.assertExpectedRaisesInline(
            RuntimeError, lambda: torch._C._nn.upsample_nearest2d(x, (16, 16), out=out),
            """Expected out tensor to have device meta, but got cpu instead"""
        )

    def test_add_meta_scalar(self):
        # From https://github.com/pytorch/pytorch/issues/53815
        x = torch.empty(2, device='meta')
        y = x + 2
        self.assertEqual(y.size(), x.size())

    def test_normal_shape(self):
        warned = False
        for device in get_all_device_types():
            tensor1 = torch.rand(1, device=device)
            tensor4 = torch.rand(4, device=device)
            tensor120 = torch.rand(120, device=device)
            tensor2145 = torch.rand(2, 1, 4, 5, device=device)
            tensor2345 = torch.rand(2, 3, 4, 5, device=device)
            tensor2345_non_contiguous = torch.rand(2, 4, 3, 5, device=device).permute(0, 2, 1, 3)
            tensor2345_channels_last = tensor2345.contiguous(memory_format=torch.channels_last)
            output2345 = torch.zeros(2, 3, 4, 5, device=device)
            output345 = torch.zeros(3, 4, 5, device=device)

            # inputs have same size
            self.assertEqual(torch.normal(tensor2345, tensor2345).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(tensor2345_non_contiguous, tensor2345).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(tensor2345, tensor2345_channels_last).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(tensor2345_non_contiguous, tensor2345_channels_last).size(), (2, 3, 4, 5))

            # scalar case
            self.assertEqual(torch.normal(tensor2345, 2).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(2, tensor2345).size(), (2, 3, 4, 5))

            # inputs are expandable tensors
            self.assertEqual(torch.normal(tensor2345, tensor1).size(), (2, 3, 4, 5))
            self.assertEqual(torch.normal(tensor2145, tensor2345).size(), (2, 3, 4, 5))

            # inputs are non-expandable tensors, but they have same number of elements
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"The size of tensor a \(120\) must match the size of "
                    r"tensor b \(5\) at non-singleton dimension 3"):
                self.assertEqual(torch.normal(tensor120, tensor2345).size(), (120,))
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"The size of tensor a \(5\) must match the size of "
                    r"tensor b \(120\) at non-singleton dimension 3"):
                self.assertEqual(torch.normal(tensor2345, tensor120).size(), (2, 3, 4, 5))

            # inputs are non-expandable tensors and they don't have same number of elements
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"The size of tensor a \(5\) must match the size of "
                    r"tensor b \(4\) at non-singleton dimension 3"):
                torch.normal(tensor2345, tensor4)

            # output and inputs are size compatible
            self.assertEqual(torch.normal(tensor2345, tensor2345, out=output2345).size(), (2, 3, 4, 5))

            # output and inputs are not size compatible
            with self.assertWarnsRegex(
                    UserWarning,
                    "This behavior is deprecated, and in a future PyTorch "
                    "release outputs will not be resized unless they have "
                    "zero elements"):
                self.assertEqual(torch.normal(tensor2345, tensor2145, out=output345).size(), (2, 3, 4, 5))
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"The size of tensor a \(5\) must match the size of "
                    r"tensor b \(120\) at non-singleton dimension 3"):
                # inputs are not expandable, output size is not the same as mean
                torch.normal(tensor2345, tensor120, out=output345)

    def test_tensoriterator_output_setup(self):
        # Test whether the output's memory layout is correct
        def test_memory_layout(x, y, scale, zero_point, out):
            self.assertEqual(x.dim(), 4)
            self.assertEqual(x.size(), y.size())
            self.assertEqual(y.size(), out.size())

            shape = x.size()
            for n in range(shape[0]):
                for c in range(shape[1]):
                    for h in range(shape[2]):
                        for w in range(shape[3]):
                            if scale is not None and zero_point is not None:
                                self.assertEqual(
                                    out[n][c][h][w],
                                    torch.ops.quantized.add(x[n][c][h][w], y[n][c][h][w], scale, zero_point))
                            else:
                                self.assertEqual(out[n][c][h][w], x[n][c][h][w] + y[n][c][h][w])

        xraw = torch.rand(2, 3, 4, 4)
        yraw = torch.rand(2, 3, 4, 4)
        qxraw = torch.quantize_per_tensor(xraw, 0.1, 5, torch.quint8)
        qyraw = torch.quantize_per_tensor(yraw, 0.1, 5, torch.quint8)

        # contiguous case fast setup
        test_memory_layout(xraw, yraw, None, None, xraw + yraw)
        test_memory_layout(qxraw, qyraw, 0.1, 5, torch.ops.quantized.add(qxraw, qyraw, 0.1, 5))

        # channels last case fast setup
        x = xraw.contiguous(memory_format=torch.channels_last)
        y = yraw.contiguous(memory_format=torch.channels_last)
        test_memory_layout(x, y, None, None, x + y)
        qx = qxraw.contiguous(memory_format=torch.channels_last)
        qy = qyraw.contiguous(memory_format=torch.channels_last)
        test_memory_layout(qx, qy, 0.1, 5, torch.ops.quantized.add(qx, qy, 0.1, 5))

        # non contiguous case fast setup (dense, non-overlapping, same shape and strides)
        x = xraw.permute(0, 2, 3, 1)
        y = yraw.permute(0, 2, 3, 1)
        test_memory_layout(x, y, None, None, x + y)
        qx = qxraw.permute(0, 2, 3, 1)
        qy = qyraw.permute(0, 2, 3, 1)
        test_memory_layout(qx, qy, 0.1, 5, torch.ops.quantized.add(qx, qy, 0.1, 5))

        # non contiguous case fast setup (dense, non-overlapping)
        # input tensors have same shape and strides
        # output tensor have same shape as input tensors but different stride
        # output tensor should preserve its strides in this case
        x = xraw.permute(0, 2, 3, 1)
        y = yraw.permute(0, 2, 3, 1)
        out = torch.empty_like(xraw)
        out = out.permute(0, 3, 2, 1)
        expected_stride = out.stride()
        test_memory_layout(x, y, None, None, torch.add(x, y, out=out))
        self.assertEqual(expected_stride, out.stride())

        # non contiguous case non fast setup
        x = xraw.permute(0, 2, 3, 1)
        y = yraw.permute(0, 3, 2, 1)
        test_memory_layout(x, y, None, None, x + y)
        qx = qxraw.permute(0, 2, 3, 1)
        qy = qyraw.permute(0, 3, 2, 1)
        test_memory_layout(qx, qy, 0.1, 5, torch.ops.quantized.add(qx, qy, 0.1, 5))

    # Tests to make sure we still handle .data properly until it is removed
    def test_dot_data_use(self):
        # .data allows to change the Tensors types inplace, check that we still
        # raise a nice error.
        with self.assertRaisesRegex(
                RuntimeError,
                # message includes both Double and ComplexFloat
                '(?=.*Double)(?=.*ComplexFloat)'):

            # Calls model with a LongTensor input but DoubleTensor weights
            input = torch.randn(1, 1, 1, 6, dtype=torch.double)
            weight = torch.zeros(1, 1, 1, 3, dtype=torch.complex64)
            model = torch.nn.Conv2d(1, 1, (1, 3), stride=1, padding=0, bias=False)
            model.weight.data = weight
            out = model(input)

    def test_empty_storage_view(self):
        # we should be able to "modify" slices of a 0-element
        # array without an error being raised due to
        # trying to resize its storage
        t = torch.from_numpy(np.empty((0, 4)))
        t[:, 1::2] *= 1

    def test_has_storage(self):
        self.assertIsNotNone(torch.tensor([]).storage())
        self.assertIsNotNone(torch.empty(0).storage())
        self.assertIsNotNone(torch.tensor([]).clone().storage())
        self.assertIsNotNone(torch.tensor([0, 0, 0]).nonzero().storage())
        self.assertIsNotNone(torch.tensor([]).new().storage())

    # FIXME: Extend this test and put in a TensorProperties test class
    def test_numel(self):
        b = torch.ByteTensor(3, 100, 100)
        self.assertEqual(b.nelement(), 3 * 100 * 100)
        self.assertEqual(b.numel(), 3 * 100 * 100)

    # Verifies that (deep)copies of dtypes are the same objects
    def test_copy_dtypes(self):
        for dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
            copied_dtype = copy.deepcopy(dtype)
            self.assertIs(dtype, copied_dtype)

    def test_dtype_is_signed(self):
        for dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.half):
            self.assertEqual(dtype.is_signed, torch.is_signed(torch.tensor(0, dtype=dtype)))

        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.quint8.is_signed)
        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.qint8.is_signed)
        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.qint32.is_signed)

    # FIXME: Put the following random tests into their own test class or test suite
    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_RNGState(self):
        state = torch.get_rng_state()
        stateCloned = state.clone()
        before = torch.rand(1000)

        self.assertEqual(state.ne(stateCloned).long().sum(), 0, atol=0, rtol=0)

        torch.set_rng_state(state)
        after = torch.rand(1000)
        self.assertEqual(before, after, atol=0, rtol=0)

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_RNGStateAliasing(self):
        # Fork the random number stream at this point
        gen = torch.Generator()
        gen.set_state(torch.get_rng_state())
        self.assertEqual(gen.get_state(), torch.get_rng_state())

        target_value = torch.rand(1000)
        # Dramatically alter the internal state of the main generator
        _ = torch.rand(100000)
        forked_value = torch.rand(1000, generator=gen)
        self.assertEqual(target_value, forked_value, atol=0, rtol=0, msg="RNG has not forked correctly.")

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_RNG_after_pickle(self):
        torch.random.manual_seed(100)
        before = torch.rand(10)

        torch.random.manual_seed(100)
        buf = io.BytesIO()
        tensor = torch.tensor([1, 2, 3])
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(tensor)
        after = torch.rand(10)

        self.assertEqual(before, after, atol=0, rtol=0)

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_boxMullerState(self):
        torch.manual_seed(123)
        odd_number = 101
        seeded = torch.randn(odd_number)
        state = torch.get_rng_state()
        midstream = torch.randn(odd_number)
        torch.set_rng_state(state)
        repeat_midstream = torch.randn(odd_number)
        torch.manual_seed(123)
        reseeded = torch.randn(odd_number)
        self.assertEqual(midstream, repeat_midstream, atol=0, rtol=0,
                         msg='get_rng_state/set_rng_state not generating same sequence of normally distributed numbers')
        self.assertEqual(seeded, reseeded, atol=0, rtol=0,
                         msg='repeated calls to manual_seed not generating same sequence of normally distributed numbers')

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_manual_seed(self):
        rng_state = torch.get_rng_state()
        torch.manual_seed(2)
        x = torch.randn(100)
        self.assertEqual(torch.initial_seed(), 2)
        torch.manual_seed(2)
        y = torch.randn(100)
        self.assertEqual(x, y)

        max_int64 = 0x7fff_ffff_ffff_ffff
        min_int64 = -max_int64 - 1
        max_uint64 = 0xffff_ffff_ffff_ffff
        # Check all boundary cases of valid seed value inputs
        test_cases = [
            # (seed, expected_initial_seed)
            # Positive seeds should be unchanged
            (max_int64, max_int64),
            (max_int64 + 1, max_int64 + 1),
            (max_uint64, max_uint64),
            (0, 0),
            # Negative seeds wrap around starting from the largest seed value
            (-1, max_uint64),
            (min_int64, max_int64 + 1)
        ]
        for seed, expected_initial_seed in test_cases:
            torch.manual_seed(seed)
            actual_initial_seed = torch.initial_seed()
            msg = "expected initial_seed() = %x after calling manual_seed(%x), but got %x instead" % (
                expected_initial_seed, seed, actual_initial_seed)
            self.assertEqual(expected_initial_seed, actual_initial_seed, msg=msg)
        for invalid_seed in [min_int64 - 1, max_uint64 + 1]:
            with self.assertRaisesRegex(RuntimeError, r'Overflow when unpacking long'):
                torch.manual_seed(invalid_seed)

        torch.set_rng_state(rng_state)

    # FIXME: Describe this test and port to the generic device framework in a more
    #   appropriate test suite for the copy operation
    def test_copy_transpose(self):
        x = torch.arange(100 * 100, dtype=torch.float).reshape(100, 100).t()
        y = torch.empty(100, 100, dtype=torch.float)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

        y = torch.empty(100, 100, dtype=torch.double)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

        # Validates regression reported in https://github.com/pytorch/pytorch/issues/45269
        x = torch.arange(100 * 100).reshape(100, 100).to(dtype=torch.cfloat).t()
        y = torch.empty(100, 100, dtype=torch.cfloat)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

        x = torch.arange(100 * 100).reshape(100, 100).to(dtype=torch.complex32).t()
        y = torch.empty(100, 100, dtype=torch.complex32)
        y.copy_(x)
        self.assertEqual(y[:, 0], range(100))
        self.assertEqual(y[:, 40], range(4000, 4100))

    # FIXME: Port to a more appropriate test suite
    @skipIfTorchInductor("FIXME")
    def test_copy_broadcast(self):
        torch.zeros(5, 6).copy_(torch.zeros(6))
        self.assertRaises(RuntimeError, lambda: torch.zeros(5, 6).copy_(torch.zeros(30)))

    # FIXME: Port to a more appropriate test suite
    def test_copy_many_to_one(self):
        # Testing in-place copy where it attempt to write from many memory
        # storage to a single storage would cause RuntimeError to be thrown
        self.assertRaises(RuntimeError, lambda: torch.zeros(1, 6).expand(5, 6).copy_(torch.zeros(5, 6)))

    @skipIfTorchInductor("FIXME")
    def test_copy_float16(self):
        # Check that fbgemm code no longer reads memory out of bounds, see
        # copy_impl and fbgemm::Float16ToFloat_ref.
        # https://github.com/pytorch/pytorch/issues/88543

        # Types to test different code paths in copy_impl.
        dtypes = (
            # out_dtype, src_dtype
            (torch.float32, torch.float16),  # fbgemm
            (torch.float16, torch.float32),  # fbgemm
            (torch.float32, torch.float32),  # TensorIterator
        )

        cases = (
            # out_shape, src_shape, is_ok
            # These cases used to crash with fbgemm, make sure these also raise
            # exceptions with TensorIterator.
            ((1, 2, 3), (0, 2, 3), False),  # same strides, not allowed by TI
            ((1, 5, 6), (4, 5, 6), False),  # same strides, not allowed by TI
            (1, (0, 2, 3), False),  # different strides
            ((4, 5, 6), (0, 2, 3), False),  # different strides
            ((4, 5, 6), (1, 2, 3), False),  # different strides
            ((4, 5, 6), (6, 5, 4), False),  # same numel

            # These cases should pass with fbgemm and TensorIterator.
            ((4, 5, 6), (1, 5, 6), True),  # same strides
            ((4, 5, 6), (4, 5, 6), True),  # same strides
            ((0, 2, 3), 1, True),  # different strides, allowed by TI
            ((4, 5, 6), (4, 5, 1), True),  # different strides, allowed by TI
        )

        for (out_shape, src_shape, is_ok), (out_dtype, src_dtype) in itertools.product(cases, dtypes):
            out = torch.zeros(out_shape, dtype=out_dtype, device=torch.device('cpu'))
            src = torch.ones(src_shape, dtype=src_dtype, device=torch.device('cpu'))
            if is_ok:
                if torch.cuda.is_available():
                    out_cuda = out.cuda()
                    src_cuda = src.cuda()
                res = out.copy_(src)
                if torch.cuda.is_available():
                    res_cuda = out_cuda.copy_(src_cuda)
                    self.assertEqual(res, res_cuda)
            else:
                self.assertRaises(RuntimeError, lambda: out.copy_(src))

    # FIXME: Port to a more appropriate test suite
    def _test_to_with_layout(self, layout):
        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True))

            devices = [t.device]
            if t.device.type == 'cuda':
                if t.device.index == -1:
                    devices.append('cuda:{}'.format(torch.cuda.current_device()))
                elif t.device.index == torch.cuda.current_device():
                    devices.append('cuda')
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True))

        a = torch.tensor(5)
        if layout == torch.sparse_csr:
            a = torch.tensor([[0, 1, 2], [2, 0, 3]]).to_sparse_csr()
        test_copy_behavior(a)
        self.assertEqual(a.device, a.to('cpu').device)
        self.assertEqual(a.device, a.to('cpu', dtype=torch.float32).device)
        self.assertIs(torch.float32, a.to('cpu', dtype=torch.float32).dtype)
        self.assertEqual(a.device, a.to(torch.float32).device)
        self.assertIs(torch.float32, a.to(dtype=torch.float32).dtype)

        def test_data_ptr(getter):
            self.assertEqual(getter(a), getter(a.to('cpu')))
            self.assertEqual(getter(a), getter(a.to(dtype=a.dtype, device=a.device, copy=False)))
            self.assertEqual(getter(a), getter(a.to('cpu', copy=False)))
            self.assertNotEqual(getter(a), getter(a.to('cpu', copy=True)))
        if layout == torch.sparse_csr:
            # TODO: compressed sparse tensors currently don't support data_ptr.
            # Exercising failure will allow us to widen coverage of this test once it does.
            with self.assertRaisesRegex(RuntimeError, "Cannot access data pointer of Tensor that doesn't have storage"):
                a.data_ptr()
            # While compressed sparse tensors don't have a concept of data_ptr
            # the underlying tensors do. The implementation of to appropriately forwards
            # the call to the components, which is what we're test here.
            test_data_ptr(lambda a: a.values().data_ptr())
            test_data_ptr(lambda a: a.crow_indices().data_ptr())
            test_data_ptr(lambda a: a.col_indices().data_ptr())
        else:
            test_data_ptr(lambda a: a.data_ptr())

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = torch.tensor(5., device=cuda)
                    test_copy_behavior(b, non_blocking)
                    self.assertEqual(b.device, b.to(cuda, non_blocking=non_blocking).device)
                    self.assertEqual(a.device, b.to('cpu', non_blocking=non_blocking).device)
                    self.assertEqual(b.device, a.to(cuda, non_blocking=non_blocking).device)
                    self.assertIs(torch.int32, b.to('cpu', dtype=torch.int32, non_blocking=non_blocking).dtype)
                    self.assertEqual(a.device, b.to('cpu', dtype=torch.int32, non_blocking=non_blocking).device)
                    self.assertIs(torch.int32, b.to(dtype=torch.int32).dtype)
                    self.assertEqual(b.device, b.to(dtype=torch.int32).device)

    @skipIfTorchInductor("FIXME")
    def test_to(self):
        self._test_to_with_layout(torch.strided)
        is_cuda10_2_or_higher = (
            (torch.version.cuda is not None)
            and ([int(x) for x in torch.version.cuda.split(".")] >= [10, 2]))
        if is_cuda10_2_or_higher:  # in cuda10_1 sparse_csr is beta
            self._test_to_with_layout(torch.sparse_csr)

    # FIXME: describe this test
    def test_as_subclass(self):
        class SubTensor(torch.Tensor):
            member_var = object()

        t0 = torch.tensor(0)
        t1 = torch.tensor([1, 2])
        t2 = torch.tensor([[3, 4], [5, 6]])

        s0 = t0.as_subclass(SubTensor)
        s1 = t1.as_subclass(SubTensor)
        s2 = t2.as_subclass(SubTensor)

        # Check that the correct type is returned.
        self.assertTrue(type(s0) is SubTensor)
        self.assertTrue(type(s1) is SubTensor)
        self.assertTrue(type(s2) is SubTensor)

        # Check that the data is equal.
        self.assertEqual(t0, s0)
        self.assertEqual(t1, s1)
        self.assertEqual(t2, s2)

        t0[()] = 1
        t1[1] = 3
        t2[1, 1] = 7

        # Check that the data is equal even after modification.
        self.assertEqual(t0, s0)
        self.assertEqual(t1, s1)
        self.assertEqual(t2, s2)

        # Check that member variables are passed through.
        self.assertTrue(s0.member_var is SubTensor.member_var)
        self.assertTrue(s1.member_var is SubTensor.member_var)
        self.assertTrue(s2.member_var is SubTensor.member_var)

        # Test that autograd is propagated.
        t = torch.tensor(5, dtype=torch.float32, requires_grad=True)

        # Run a calculation on the tensor.
        exp_t = torch.exp(t)

        # Cast exp_t to a subclass.
        exp_s = exp_t.as_subclass(SubTensor)

        # Make sure that t.grad was initially None
        self.assertTrue(t.grad is None)

        # Run the autograd calculation.
        exp_s.backward()

        # Make sure autograd was propagated to the original tensor
        # declared with requires_grad.
        self.assertTrue(t.grad is not None)

        # Make sure invalid subclasses raise nice errors
        class BadSubTensor():
            member_var = object()

        err_msg = "Creating a Tensor subclass from a class that does not inherit from Tensor"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            s0 = t0.as_subclass(BadSubTensor)

    # FIXME: Port to a test suite that better fits slicing
    @skipIfTorchInductor("FIXME")
    def test_slice(self):
        empty = torch.empty(0, 4)
        x = torch.arange(0., 16).view(4, 4)
        self.assertEqual(x[:], x)
        self.assertEqual(x[:4], x)
        # start and stop are clamped to the size of dim
        self.assertEqual(x[:5], x)
        # if start >= stop then the result is empty
        self.assertEqual(x[2:1], empty)
        self.assertEqual(x[2:2], empty)
        # out of bounds is also empty
        self.assertEqual(x[10:12], empty)
        # additional correctness checks
        self.assertEqual(x[:1].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(x[:-3].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(x[:, -2:3].tolist(), [[2], [6], [10], [14]])
        self.assertEqual(x[0:-1:2].tolist(), [[0, 1, 2, 3], [8, 9, 10, 11]])

    def test_type(self):
        x = torch.randn(3, 3).double()
        self.assertEqual(x.type('torch.FloatTensor').dtype, torch.float32)
        self.assertEqual(x.type(torch.FloatTensor).dtype, torch.float32)
        self.assertEqual(x.int().type(torch.Tensor).dtype, torch.get_default_dtype())
        self.assertEqual(x.type(torch.int32).dtype, torch.int32)

    # FIXME: port to a quantization test suite
    def test_qengine(self):
        qengines = torch.backends.quantized.supported_engines
        original_qe = torch.backends.quantized.engine
        for qe in qengines:
            torch.backends.quantized.engine = qe
            assert torch.backends.quantized.engine == qe, 'qengine not set successfully'
        torch.backends.quantized.engine = original_qe

    # FIXME: port to a distributed test suite -- also... how could this be OOMing on Windows CUDA?
    @slowTest
    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "Disabled for environments that \
                        don't support multiprocessing with spawn start method")
    @unittest.skipIf(IS_WINDOWS, 'FIXME: CUDA OOM error on Windows')
    def test_multinomial_invalid_probs(self):
        def _spawn_method(self, method, arg):
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            with mp.Pool(1) as pool:
                out: list = pool.map(method, [arg])
                self.assertTrue(out[0])

        def _test_multinomial_invalid_probs(probs):
            try:
                # n_sample = 1 is a special case, test n_sample=2 which is more general
                torch.multinomial(probs.to('cpu'), 2)
                return False  # Should not be reached
            except RuntimeError as e:
                return 'probability tensor contains either `inf`, `nan` or element < 0' in str(e)

            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., -1., 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., inf, 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., -inf, 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., 1., nan]))

    # FIXME: port to more appropriate test suite
    def test_to_with_tensor(self):
        a = torch.tensor(5)
        self.assertEqual(a.device, a.to(a).device)

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = torch.tensor(5., device=cuda)
                    self.assertEqual(b.device, b.to(b, non_blocking=non_blocking).device)
                    self.assertEqual(a.device, b.to(a, non_blocking=non_blocking).device)
                    self.assertEqual(b.device, a.to(b, non_blocking=non_blocking).device)

    def test_device(self):
        cpu = torch.device('cpu')
        self.assertEqual('cpu', str(cpu))
        self.assertEqual('cpu', cpu.type)
        self.assertEqual(None, cpu.index)

        cpu0 = torch.device('cpu:0')
        self.assertEqual('cpu:0', str(cpu0))
        self.assertEqual('cpu', cpu0.type)
        self.assertEqual(0, cpu0.index)

        cpu0 = torch.device('cpu', 0)
        self.assertEqual('cpu:0', str(cpu0))
        self.assertEqual('cpu', cpu0.type)
        self.assertEqual(0, cpu0.index)

        cuda = torch.device('cuda')
        self.assertEqual('cuda', str(cuda))
        self.assertEqual('cuda', cuda.type)
        self.assertEqual(None, cuda.index)

        cuda1 = torch.device('cuda:1')
        self.assertEqual('cuda:1', str(cuda1))
        self.assertEqual('cuda', cuda1.type)
        self.assertEqual(1, cuda1.index)

        cuda1 = torch.device('cuda', 1)
        self.assertEqual('cuda:1', str(cuda1))
        self.assertEqual('cuda', cuda1.type)
        self.assertEqual(1, cuda1.index)

        cuda90 = torch.device('cuda', 90)
        self.assertEqual('cuda:90', str(cuda90))
        self.assertEqual('cuda', cuda90.type)
        self.assertEqual(90, cuda90.index)

        self.assertRaises(RuntimeError, lambda: torch.device('cpu:-1'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:-1'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 '))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda: 2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2.'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2?'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:?2'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2.232'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2 cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2+cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device('cuda:2cuda:3'))
        self.assertRaises(RuntimeError, lambda: torch.device(-1))

        self.assertRaises(RuntimeError, lambda: torch.device('other'))
        self.assertRaises(RuntimeError, lambda: torch.device('other:0'))

        device_set = {'cpu', 'cpu:0', 'cuda', 'cuda:0', 'cuda:1', 'cuda:10', 'cuda:100'}
        device_hash_set = set()
        for device in list(device_set):
            device_hash_set.add(hash(torch.device(device)))
        self.assertEqual(len(device_set), len(device_hash_set))

        def get_expected_device_repr(device):
            if device.index is not None:
                return "device(type='{type}', index={index})".format(
                    type=device.type, index=device.index)

            return "device(type='{type}')".format(type=device.type)

        for device in device_set:
            dev = torch.device(device)
            self.assertEqual(repr(dev), get_expected_device_repr(dev))

    # Tests that the use_deterministic_flag can be set as expected
    @wrapDeterministicFlagAPITest
    def test_deterministic_flag(self):
        for deterministic, warn_only in product([True, False], [True, False]):
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
            self.assertEqual(warn_only, torch.is_deterministic_algorithms_warn_only_enabled())

            if deterministic:
                if warn_only:
                    debug_mode = 1
                else:
                    debug_mode = 2
            else:
                debug_mode = 0

            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())

        for debug_mode in [0, 1, 2]:
            torch.set_deterministic_debug_mode(debug_mode)
            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())
            deterministic = debug_mode in [1, 2]
            warn_only = debug_mode == 1

            self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
            self.assertEqual(warn_only, torch.is_deterministic_algorithms_warn_only_enabled())

        for debug_mode, debug_mode_str in [(0, 'default'), (1, 'warn'), (2, 'error')]:
            torch.set_deterministic_debug_mode(debug_mode_str)
            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())

        with self.assertRaisesRegex(
                TypeError,
                r"_set_deterministic_algorithms\(\): argument 'mode' \(position 1\) must be bool, not int"):
            torch.use_deterministic_algorithms(1)

        with self.assertRaisesRegex(
                TypeError,
                r"_set_deterministic_algorithms\(\): argument 'warn_only' must be bool, not int"):
            torch.use_deterministic_algorithms(False, warn_only=1)

    def test_type_conversion_via_dtype_name(self):
        x = torch.tensor([1])
        self.assertEqual(x.byte().dtype, torch.uint8)
        self.assertEqual(x.bool().dtype, torch.bool)
        self.assertEqual(x.char().dtype, torch.int8)
        self.assertEqual(x.double().dtype, torch.float64)
        self.assertEqual(x.float().dtype, torch.float32)
        self.assertEqual(x.half().dtype, torch.float16)
        self.assertEqual(x.int().dtype, torch.int32)
        self.assertEqual(x.bfloat16().dtype, torch.bfloat16)
        cfloat = x.cfloat()
        self.assertEqual(cfloat.dtype, torch.complex64)
        self.assertEqual(cfloat.real, x.float())
        self.assertEqual(cfloat.imag, torch.zeros_like(cfloat.imag))
        cdouble = x.cdouble()
        self.assertEqual(cdouble.dtype, torch.complex128)
        self.assertEqual(cdouble.real, x.double())
        self.assertEqual(cdouble.imag, torch.zeros_like(cdouble.imag))
        chalf = x.chalf()
        self.assertEqual(chalf.dtype, torch.complex32)
        self.assertEqual(chalf.real, x.half())
        self.assertEqual(chalf.imag, torch.zeros_like(chalf.imag))

    def test_type_alias(self):
        type_alias_map = {torch.float64: torch.double,
                          torch.float32: torch.float,
                          torch.int32: torch.int,
                          torch.int64: torch.long,
                          torch.int16: torch.short,
                          torch.float16: torch.half,
                          torch.complex32: torch.chalf,
                          torch.complex64: torch.cfloat}
        for dtype, alias in type_alias_map.items():
            self.assertIs(alias, dtype)

    def test_doc_template(self) -> None:
        """
        Test that all public API doc strings use the same standard template for
        all common arguments such as tensor or dim
        """
        from torch._torch_docs import __file__ as doc_file
        from torch._torch_docs import multi_dim_common, single_dim_common, factory_common_args, factory_like_common_args

        with open(doc_file, "r", encoding="utf-8") as f:
            doc_strs = f.read()

        matches = re.findall(
            r'add_docstr\(([^,]+?),[^"\']*?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')(?:\.|,?[^,\)]*?\))',
            doc_strs,
            re.MULTILINE | re.DOTALL,
        )
        self.assertTrue(matches)

        for m in matches:
            func = m[0].strip()
            desc = m[1].strip()

            for common_args in [multi_dim_common, single_dim_common, factory_common_args, factory_like_common_args]:
                for k, v in common_args.items():
                    self.assertNotIn(v, desc, 'The argument description "{}" in {} can be '
                                              'replaced by {{{}}}'.format(v, func, k))

    def test_doc(self):
        checked_types = (types.MethodType, types.FunctionType,
                         types.BuiltinFunctionType, types.BuiltinMethodType)

        def _test_namespace(ns, *skips):
            if isinstance(ns, object):
                ns_name = ns.__class__.__name__
            else:
                ns_name = ns.__name__
            skip_regexes = []
            for r in skips:
                if isinstance(r, string_classes):
                    skip_regexes.append(re.compile('^{}$'.format(re.escape(r))))
                else:
                    skip_regexes.append(r)

            for name in dir(ns):
                if name.startswith('_'):
                    continue
                if name in ['real', 'imag']:
                    y = torch.randn(1, dtype=torch.cfloat)
                    var = getattr(y, name)
                elif name in ["H", "mT", "mH"]:
                    y = torch.randn(1, 1)
                    var = getattr(y, name)
                else:
                    var = getattr(ns, name)
                if not isinstance(var, checked_types):
                    continue
                doc = var.__doc__
                has_doc = doc is not None and len(doc.strip()) > 0
                full_name = ns_name + '.' + name
                if any(r.match(name) for r in skip_regexes):
                    self.assertFalse(has_doc,
                                     'New docs have been added for {}, please remove '
                                     'it from the skipped list in TestTorch.test_doc'.format(full_name))
                else:
                    self.assertTrue(has_doc, '{} is missing documentation'.format(full_name))

            # FIXME: All of the following should be marked as expected failures
            # so that it is easier to tell when missing has been added.
            # FIXME: fix all the skipped ones below!
            test_namespace(torch.randn(1),
                           'as_strided_',
                           re.compile('^clamp_(min|max)_?$'),
                           'is_distributed',
                           'is_nonzero',
                           'is_same_size',
                           'log_softmax',
                           'map2_',
                           'new',
                           'reinforce',
                           'relu',
                           'relu_',
                           'prelu',
                           'resize',
                           'resize_as',
                           'softmax',
                           'split_with_sizes',
                           'unsafe_split_with_sizes',
                           '_autocast_to_fp16',
                           '_autocast_to_fp32',
                           )

            test_namespace(torch.nn)
            test_namespace(torch.nn.functional, 'assert_int_or_pair')
            # TODO: add torch.* tests when we have proper namespacing on ATen functions
            # test_namespace(torch)

    # FIXME: deprecate torch.Tensor constructor
    def test_tensor_ctor_scalar(self):
        x = torch.Tensor(torch.tensor(1.0))
        self.assertEqual(x, torch.tensor(1.0))

    def test_deepcopy_gradient(self):
        from copy import deepcopy
        a = torch.zeros(10)
        a.grad = torch.ones(10)
        self.assertEqual(a.grad, deepcopy(a).grad)
        s = torch.zeros(10).to_sparse()
        s.grad = torch.ones(10).to_sparse()
        self.assertEqual(s.grad, deepcopy(s).grad)

        # ensure sharing is not broken
        c = deepcopy([a, a.grad])
        self.assertTrue(c[0].grad is c[1])

    def test_tensor_base_init(self):
        # Direct construction not OK
        self.assertRaises(RuntimeError, lambda: torch._C._TensorBase())

        # But construction of subclass is OK
        class T(torch._C._TensorBase):
            pass

        T()

    def test_tensor_base_new(self):

        # OK to call super().__new__, see
        # https://github.com/pytorch/pytorch/issues/57421
        class TestTensor(torch._C._TensorBase):
            @staticmethod
            def __new__(cls, x, *args, **kwargs):
                return super().__new__(cls, x, *args, **kwargs)

        x = torch.ones(5)
        test_tensor = TestTensor(x)

    def test_pyobj_preserved(self):
        x = torch.empty(2)
        x.foo = 2  # put something on __dict__
        y = torch.empty(2)
        y.grad = x
        del x  # x is dead in Python
        self.assertEqual(y.grad.foo, 2)
        z = y.grad  # it's live
        del z  # it's dead again
        self.assertEqual(y.grad.foo, 2)

    def test_subclass_preserved(self):
        class MyTensor(torch.Tensor):
            pass

        x = MyTensor(torch.empty(2))
        y = torch.empty(2)
        y.grad = x
        del x  # x is dead in Python
        self.assertEqual(type(y.grad), MyTensor)
        z = y.grad  # it's live
        del z  # it's dead again
        self.assertEqual(type(y.grad), MyTensor)

    def test_tensor_slot_dealloc(self):

        class SlotTensor1(torch._C._TensorBase):
            __slots__ = ['slot1']

        class SlotTensor2(SlotTensor1):
            __slots__ = ['slot2']

        m1, t1 = Tracker.make()
        m2, t2 = Tracker.make()
        slot_tensor = SlotTensor2(torch.empty(2))
        slot_tensor.slot1 = t1
        slot_tensor.slot2 = t2
        del t1
        del t2
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])
        del slot_tensor
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_tensor_dict_dealloc(self):
        m, t = Tracker.make()
        x = torch.empty(2)
        x.arf = t
        del t
        self.assertFalse(m[0])
        del x
        self.assertTrue(m[0])

    def test_tensor_finalizer_dealloc(self):
        m = [False]

        class FinalizerTensor(torch._C._TensorBase):
            def __del__(self):
                m[0] = True

        fin_tensor = FinalizerTensor(torch.empty(2))
        self.assertFalse(m[0])
        del fin_tensor
        self.assertTrue(m[0])

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_tensor_weakref_dealloc(self):

        x = torch.empty(2)
        m = [False]

        def cb(r):
            m[0] = True

        wref = weakref.ref(x, cb)
        del x
        self.assertTrue(m[0])
        self.assertEqual(wref(), None)

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_tensor_cycle_via_dict(self):
        m1, t1 = Tracker.make()
        x = torch.empty(2)
        x._tracker = t1
        del t1

        m2, t2 = Tracker.make()
        y = torch.empty(2)
        y._tracker = t2
        del t2

        x._loop = y
        y._loop = x

        # C++ reference should keep the cycle live!
        # This exercise THPVariable_subtype_traverse
        # NB: Because z.grad is a reference done entirely in C++, cycles
        # involving it directly are NOT broken by Python GC; you've
        # set up a good old C++ reference cycle which we cannot safely
        # break (because C++ references are allowed to be accessed
        # multithreaded-ly) (TODO: except maybe if you can prove that
        # only Python has access to the C++ object, in which case you can
        # also prove that no multithreaded access occurs)
        z = torch.empty(2)
        z.grad = x

        del x
        del y

        gc.collect()
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])

        with disable_gc():
            del z
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        gc.collect()
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    def test_tensor_cycle_via_slots(self):
        m1 = [False]
        m2 = [False]

        class SlotTensor1(torch._C._TensorBase):
            __slots__ = ['slot1']

            def __del__(self):
                m1[0] = True

        class SlotTensor2(SlotTensor1):
            __slots__ = ['slot2']

            def __del__(self):
                m2[0] = True

        x = SlotTensor1(torch.empty(2))
        y = SlotTensor2(torch.empty(2))

        x.slot1 = y
        y.slot2 = x

        del x
        with disable_gc():
            del y
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        gc.collect()
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    # FIXME: move to test_autograd?
    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_backward_hooks_traverse(self):
        m1, t1 = Tracker.make()
        m2, t2 = Tracker.make()
        x = torch.empty(2, requires_grad=True)
        x._tracker = t1
        y = torch.empty(2, requires_grad=True)
        y._tracker = t2
        del t1
        del t2

        # this hits a special setter, it's not just a __dict__ entry
        x._backward_hooks = y
        y._backward_hooks = x

        del x
        with disable_gc():
            del y
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        gc.collect()

        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_dead_weak_ref(self):
        x = torch.empty(2)
        w_x = weakref.ref(x)
        y = torch.empty(2)
        y.grad = x
        del x

        x = w_x()
        # Ideally, x would keep the tensor live.  But CPython doesn't
        # provide enough hooks to do this.  So it will go dead and x
        # will transmute into an undefined tensor.  Not great, but the
        # best we can do.
        del y

        self.assertRaises(RuntimeError, lambda: x.sigmoid())

    def test_resurrected_weak_ref(self):
        x = torch.empty(2)
        w_x = weakref.ref(x)
        y = torch.empty(2)
        y.grad = x
        del x

        x = w_x()
        # Use this to manually fix weak references after dereferencing them
        x._fix_weakref()
        del y
        x.sigmoid()

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_fix_weakref_no_leak(self):
        import weakref

        called = False

        a = torch.randn(1)

        def callback(w):
            nonlocal called
            called = True
        wa = weakref.ref(a, callback)
        a._fix_weakref()
        del a

        self.assertTrue(called)

    # FIXME: move to test_linalg
    @torch.inference_mode()
    def test_bmm_multithreaded(self):
        device = 'cpu'
        num_threads = torch.get_num_threads()

        torch.set_num_threads(4)
        batch_sizes = [1, 10]
        M, N, O = 23, 8, 12
        dtype = torch.float32
        numpy_dtype = dtype

        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_inputs(num_batches):
            # transposed tensors
            for perm1, perm2 in itertools.product(itertools.permutations((0, 1, 2)), repeat=2):
                b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-1, high=1)
                b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-1, high=1)
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                yield b1, b2
            # broadcasting tensors
            for b1, b2, b3, b4, b5, b6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if b1 else 1, M if b2 else 1, N if b3 else 1)
                shape2 = (num_batches if b4 else 1, N if b5 else 1, O if b6 else 1)
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, M, N)
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, N, O)
                yield b1, b2
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = torch.randn(shape1, dtype=dtype, device=device)
                b2 = torch.randn(shape2, dtype=dtype, device=device)
                yield b1, b2

        try:
            for num_batches in batch_sizes:
                for (b1, b2), perm3 in itertools.product(generate_inputs(num_batches), itertools.permutations((0, 1, 2))):
                    res1 = torch.bmm(b1, b2)
                    res2 = torch.full((num_batches, M, O), math.nan, dtype=dtype, device=device) \
                        .permute(perm3).contiguous().permute(invert_perm(perm3))
                    torch.bmm(b1, b2, out=res2)
                    expect = torch.from_numpy(
                        b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                    self.assertEqual(expect, res1)
                    self.assertEqual(expect, res2)
        finally:
            torch.set_num_threads(num_threads)

    def test_conj_neg_tolist(self):
        x = torch.randn(2, dtype=torch.cfloat)
        y1 = x.conj()
        y1_expect = x.conj_physical()
        y2 = y1.imag
        self.assertEqual(y1, y1_expect.tolist())
        self.assertEqual(y2, y1_expect.imag.tolist())

    @unittest.skipIf(torch.backends.cuda.is_built(), "Skipped for cuda-enabled build")
    def test_no_cuda_monkeypatch(self):
        # Note that this is not in test_cuda.py as this whole file is skipped when cuda
        # is not available.
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate dummy base class Stream"):
            torch.cuda.Stream()

        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate dummy base class Event"):
            torch.cuda.Event()

        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate dummy base class CUDAGraph"):
            torch.cuda.graphs.CUDAGraph()

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
        # ('narrow', (10, 20, 30), lambda: [DIM_ARG, 0, 5], [METHOD]),
        # ('transpose', (10, 20, 30), lambda: [DIM_ARG, DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
        # ('size', (10, 20, 30), lambda: [DIM_ARG], [METHOD]),
        # ('cat', [(2, 3, 4), (2, 3, 4)], lambda: [DIM_ARG], [FUNCTIONAL]),
        # ('chunk', (10, 20, 30), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('gather', (10, 20), lambda: [DIM_ARG, idx_tensor((10, 20), 10)], [METHOD, FUNCTIONAL]),
        # ('index_select', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10)], [METHOD, FUNCTIONAL]),
        # ('split', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('squeeze', (10, 1, 20, 1), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
        # ('unbind', (2, 3, 4), lambda: [DIM_ARG], [FUNCTIONAL]),
        # ('unsqueeze', (10, 20), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL], 1),
        # ('logcumsumexp', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('cumprod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('cumsum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('cummax', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        ('cummin', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('mean', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('median', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('nanmedian', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('mode', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('norm', (10, 20), lambda: [2, DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('prod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('std', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('sum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('var', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('kthvalue', (10, 20), lambda: [3, DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('max', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('min', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('sort', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('topk', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
        # ('renorm', (10, 20), lambda: [2, DIM_ARG, 1], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
        # ('index_add', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
        # ('index_copy', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
        # ('index_fill', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), 12], [INPLACE_METHOD]),
        # ('scatter', (10, 10), lambda: [DIM_ARG, idx_tensor((10, 10), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
        # ('select', (10, 20), lambda: [DIM_ARG, 3], [METHOD]),
        # ('unfold', (10, 20), lambda: [DIM_ARG, 5, 2], [METHOD]),
    ]

    for decl in neg_dim_tests:
        if len(decl) == 4:
            name, tensor_arg, arg_constr, types = decl
            extra_dim = 0
        elif len(decl) == 5:
            name, tensor_arg, arg_constr, types, extra_dim = decl

        test_name = 'test_' + name + '_neg_dim'

        assert not hasattr(TestTorch, test_name), "Duplicated test name: " + test_name
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
# instantiate_device_type_tests(TestViewOps, globals())
# instantiate_device_type_tests(TestVitalSignsCuda, globals())
# instantiate_device_type_tests(TestTensorDeviceOps, globals())
instantiate_device_type_tests(TestTorchDeviceType, globals())
# instantiate_device_type_tests(TestDevicePrecision, globals(), except_for='cpu')

if __name__ == '__main__':
    run_tests()

