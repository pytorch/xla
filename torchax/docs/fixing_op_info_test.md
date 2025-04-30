# How to fix an op info test.

## What is OpInfo test

PyTorch created a list of python objects (OpInfo) to keep
track how to test each op. This is useful to us because it
ensures that the ops we implement produces the same results
pytorch would produce.

Context:
* https://dev-discuss.pytorch.org/t/opinfos-in-pytorch-1-10/253
* https://github.com/pytorch/pytorch/issues/54261


## How to fix one

### Remove one op from skiplist

Open [test/test_ops.py](../test/test_ops.py) with your
favorite text editor.
Remove one line from the `skiplist` set.

i.e.

```bash
(base) hanq-macbookpro:torchax hanq$ git diff
diff --git a/experimental/torchax/test/test_ops.py b/experimental/torchax/test/test_ops.py
index 72a39ae85..2a156cbce 100644
--- a/experimental/torchax/test/test_ops.py
+++ b/experimental/torchax/test/test_ops.py
@@ -15,7 +15,6 @@ skiplist = {
     "_native_batch_norm_legit",
     "_segment_reduce",
     "_upsample_bilinear2d_aa",
-    "addbmm",
     "addmm",
     "addmv",
     "addr",
```

### Run test to see what failure
For errors you might get after running test, there are two kind:
- Target op failure
  - error shows related to target op, such as `No lowering found for 'aten::addbmm'`, please follow instruction like [Fix Target op failure](https://github.com/pytorch/xla/blob/ManfeiBai-patch-99/experimental/torchax/docs/fixing_op_info_test.md#fix-target-op-failure)
- Decomposed op failure
  - no implementation found for target ops, but error is not `no lowering`, error shows target op has been implemented somewhere; for sitution like this, please follow instruction like [Fix Decomposed op failure](https://github.com/pytorch/xla/blob/ManfeiBai-patch-99/experimental/torchax/docs/fixing_op_info_test.md#fix-other-op-failure)

#### Fix Target op failure
Error gotten:

```
(base) hanq-macbookpro:torchax hanq$ python test/test_ops.py
...
E         RuntimeError: ('No lowering found for\n\nTo execute this test, run the following from the base repo dir:\n     python test/test_ops.py -k test_reference_eager_addbmm_cpu_int64\n\nThis message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0', 'aten::addbmm')
```

From here we have 2 strategies for fixing this test:

1. Add an implementation to `aten::addbmm` operator using Jax ops. Or,
2. Add an  implementation `aten::addbmm` operator using torch ops (this commonly known as "decompositions").

Either way works for torchax. For ops that are not "Core Aten" sometimes we implement in torch ops with the goal of
upstreaming this decomposition to [pytorch decompositon](https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py)
so other projects can benefit from it.

For illustration purposes, let's implement this op in Jax.

(NOTE: this doesn't stop us from upstreaming a decomposition later if we want)

#### Fix Decomposed op failure
For situation that no target op(`trapezoid`) implemention found in `experimental/torchax/torchax/ops/jaten.py`, but error shows target op(`trapezoid`) has been implemented somewhere:
```
======================================================================
FAIL: test_reference_eager_trapezoid_cpu_int64 (__main__.TestOpInfoCPU) [torchax_diff:0.001]
----------------------------------------------------------------------
...
AssertionError: The values for attribute 'dtype' do not match: torch.float64 != torch.float32.
```
Please try to fix it by following these steps:
  1. confirm your target op `trapezoid` is decomposed by running this code to print each sub ops:
  ```
  import torch
  import torchax

  env = torchax.default_env()
  env.config.debug_print_each_op = True
  env.config.debug_accuracy_for_each_op = True

  with env:
    y = torch.tensor([1, 5, 10])
    print(torch.trapezoid(y))
  ```
  2. (optional) Debug by modify [debug_accuracy()](https://github.com/pytorch/xla/blob/c26b19ebdefccd3a4300763e1085724d3d4cd3d0/experimental/torchax/torchax/tensor.py#L171C1-L194C14) to check `res`(from jax) and `expected_res`(from torch)'s value and dtype/type.
  3. you might need to debug/modify/add implementation of sub ops(found in step1) to support `trapezoid` by using step 2, like:
  ```
  @op(torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
  def _aten_mul(x, y):
    new_dtype = mappings.t2j_dtype(torch.get_default_dtype())
    res = x * y
    if isinstance(x, float) or isinstance(y, float):
      res = res.astype(new_dtype)
    return res
  ```

### First Impl

To implement this op using jax ops, we first find what
is the exact semantics in this page:
https://pytorch.org/docs/stable/generated/torch.addbmm.html

From it's math formula: we can implement it as follows.

```
+@op(torch.ops.aten.addbmm.default)
+def _aten_addbmm(input, batch1, batch2, *, beta=1, alpha=1):
+
+  mm = jnp.einsum('bxy, byz -> xz', batch1, batch2)
+  return beta * input + alpha * mm
```

Now running test again:

```
python test/test_ops.py -k test_reference_eager_addbmm_cpu_int64
```

(NOTE: the exact test command is printed out when we run
`pytest test/test_ops.py` so we can only run the failed test instead of running all tests.)

We now see this error:

```
FAIL: test_reference_eager_addbmm_cpu_int64 (__main__.TestOpInfoCPU) [torchax_diff:0.001]
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/test/test_ops.py", line 654, in run_export_and_compare
    diff_output(
  File "/Users/hanq/git/qihqi/torch_xla/experimental/torchax/test/test_ops.py", line 617, in diff_output
    testcase.assertTrue(
AssertionError: False is not true
```

This is telling me that our implementation did not produce
the same result as the ops in PyTorch.

To debug this, let's figure out what exact input caused this.
We can achieve this by setting a break point [here](https://github.com/pytorch/xla/blob/master/experimental/torchax/test/test_ops.py#L644), right before the diff. Here we can
inspect values of `res` and `res2`, as well as the `sample_input`.

The sample input we get is
```
SampleInput(input=tensor([[-3, -3,  9,  8, -8, -3, -4,  2,  2,  2],
        [-5,  1, -9,  9,  1, -5,  6,  1, -4, -5],
        [-2, -1,  5, -2, -3,  0,  5, -4,  9, -6],
        [-1, -7,  6,  3,  8,  3,  8,  9, -5,  7],
        [-3, -4, -9,  9,  7, -3, -8,  2,  5, -3]]), args=(tensor([[[-2,  4, -2,  5,  8],
         [-6, -2,  5,  7,  7],
         [-8, -3,  2,  5, -3],
         [-4,  7,  0, -9,  8],
         [ 3,  9, -9, -2,  0]],

        [[-7,  1, -3,  7, -4],
         [ 3,  5,  4,  6,  5],
         [-2,  8,  3,  5,  7],
         [ 8, -2, -8,  2,  0],
         [ 6,  1, -8,  8,  0]],

        [[ 2, -1, -5, -8, -9],
         [ 5,  0, -4, -1, -6],
         [-6,  2, -5, -2, -5],
         [-5, -3, -5, -4,  9],
         [-3,  4, -9, -9,  7]],

        [[ 2,  5, -7, -3,  8],
         [-5, -7, -8, -4,  4],
         [-4, -6, -3,  0,  6],
         [ 8,  0, -3, -8,  2],
         [-4,  3, -9, -6,  7]],

        [[ 2,  1, -6,  2,  8],
         [ 2,  6,  4,  1,  8],
         [-9,  9, -5,  8,  3],
         [-5,  0, -2,  4,  0],
         [ 5,  8, -4,  9,  7]]]), tensor([[[-1, -8,  3,  5, -8,  2, -5,  0, -9, -5],
         [-4, -7,  2,  2,  1, -9,  2,  7, -1, -1],
         [ 1,  8, -6, -4, -6, -8, -7, -9,  7,  4],
         [-4,  1, -9,  3,  4,  6,  0, -2, -2, -7],
         [ 5,  5,  0,  8, -3,  7, -7,  8,  3,  5]],

        [[ 8, -4, -9,  9,  5,  0,  5,  0, -5,  5],
         [-5, -3, -2,  8,  1, -2,  4, -7,  5,  3],
         [-4,  4,  1, -4, -8,  2, -5,  2,  9, -7],
         [ 9,  6, -8, -3,  3,  1,  4,  6, -5, -4],
         [-2,  1,  5,  5,  2,  6,  7, -3, -7,  3]],

        [[ 9, -8,  5, -3, -1,  2, -9, -5, -1, -3],
         [-3,  3, -9, -7, -9, -8,  1, -3,  7, -2],
         [ 8, -1,  8, -8, -7,  4,  8,  8,  5, -7],
         [-1,  6, -8,  7, -1, -5, -8,  6, -2,  8],
         [-5, -5,  8,  6,  0,  1,  3, -2, -3, -9]],

        [[ 7, -2,  6, -8, -5,  3,  2, -1, -5,  8],
         [-6, -4,  3,  9, -9, -8, -7,  3,  9,  0],
         [ 1,  3,  4,  4, -5, -2, -4, -2,  3, -7],
         [-6,  9,  5, -1,  7,  7,  8, -3, -8,  0],
         [-1, -6, -3,  3,  3, -8, -4,  9, -5,  7]],

        [[-5, -3, -9,  6, -1, -7,  9, -8,  1, -8],
         [-8, -8, -2, -5, -7, -8,  1,  0,  0, -6],
         [ 7, -5,  2,  2,  0, -9, -5, -7,  1,  8],
         [-4,  0,  9,  6, -1, -6,  6, -6, -2, -1],
         [ 7,  3,  0,  1,  1, -9,  5, -8, -1, -7]]])), kwargs={'beta': 0.6, 'alpha': 0.2}, broadcasts_input=False, name='')
```

And the `res` from torch is

```
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
```

So few observation is:
1. Input tensor are of type int64
2. alpha and beta are both floats.

So one can suspect that it has to do with rounding.
Reading the doc more carefully, we can find this sentence

    For inputs of type FloatTensor or DoubleTensor, arguments beta and alpha must be real numbers, otherwise they should be integers.

So likely torch first casted the float alpha and beta to integer, which yields 0, then used them in math to get a matrix with all zeros.

### Second Impl

```python
+@op(torch.ops.aten.addbmm.default)
+def _aten_addbmm(input, batch1, batch2, *, beta=1, alpha=1):
+  alpha = jnp.array(alpha).astype(batch1.dtype)
+  beta = jnp.array(beta).astype(batch1.dtype)
+  mm = jnp.einsum('bxy, byz -> xz', batch1, batch2)
+  return jax.lax.cond(beta == 0,
+           lambda: alpha * mm,
+           lambda: beta*input + alpha*mm)
+
```

Adding type casts makes the tests passes.

### Submit
Now, let's remove the pdb and prints we added, and submit the fix as a PR: https://github.com/pytorch/xla/pull/6993

