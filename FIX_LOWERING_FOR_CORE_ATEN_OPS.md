In order for PyTorch/XLA to support the PyTorch core ATen opset, it requires lowering each core ATen op in PyTorch/XLA. Note that this document will serve as a guide for fixing these lowering for core aten opset, specifically looking at [test_core_aten_ops.py test](https://github.com/pytorch/xla/blob/master/test/test_core_aten_ops.py). This guide will **not** cover how to lower an op in PyTorch/XLA, please refer our [op lowering guide](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md) for this. 

We also have a worklog for lowering these core aten ops in a GitHub issue, so we can track who's working on which ops and share some findings: [[Core Aten ops] Logs for fixing core aten ops coverage issues](https://github.com/pytorch/xla/issues/5934). 

Let's go back and take a closer look at the [test_core_aten_ops.py test](https://github.com/pytorch/xla/blob/master/test/test_core_aten_ops.py), which is the source of truth to verify and correctness of these lowerings. The core of this test file is the `run_export_and_compare` at https://github.com/pytorch/xla/blob/master/test/test_core_aten_ops.py#L28. Each op unit test initializes the input and passes the op as a function and its inputs. The `run_export_and_compare` has multiple subtests that have the following structure: 
- `torch_eval`
   - `torch_xla_eval`
     - `torch_xla_diff`
   - `can_export`
     - `can_convert_to_stablehlo`
       - `stablehlo_can_run`
         - `stablehlo_diff`

Below we'll describe what each of these subtests mean and give some recommendations on fixing it.

### `torch_eval`

This subtest directly calls torch version of the op with the given inputs. If the unit test fails in this subtest, this implies that torch there is a problem with the unit test itself. One common reason might be due to inputs (or types of inputs) not being compatible with the op. We recommend you to look at the official torch documentation of the corresponding op to ensure that that unit tests are passing valid inputs to the op.

### `torch_eval_xla`

This subtest calls the torch_xla version of the op. If you've made changes to lower the op and this subtest fails, this means there may be something wrong with the lowering. We recommend you to take another look at our [op lowering guide](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md). If you're unable to debug further, feel free to leave a comment in your assigned GitHub issue.

### `torch_xla_diff`

This subtest compares the output of the op between torch and torch_xla.
If this subtest fails, it implies that your lowering runs successfully 
but produced a different result than torch eager mode.

If the test uses 16-bit floats (float16, bfloat16); This is very likely
that the tolerances that we give to `torch.allclose` to compare was to 
strict. You can relax it a bit. Take a look at [this issue](https://github.com/pytorch/xla/issues/5934) of one such example.

If the result torchxla produces is totally different than what torch produces, that means it's a bug in lowering code; and probably need
more work. Feel free to tag more people (such as qihqi to look).

### `can_export`, `can_convert_to_stablehlo`, `stablehlo_can_run`, `stablehlo_diff`

These subtests are related to `export` and `stablehlo`. If the lowering is complete and the above `torch_*` subtests all succeed, it is highly likely that these tests will also succeed.  
