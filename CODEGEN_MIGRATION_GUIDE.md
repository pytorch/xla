# Codegen migration Guide

## Background
As PyTorch/XLA migrates to the LTC (Lazy Tensor Core), we need to clean up the existing stub code (which spans over 6+ files) that were used to do the op lowering. The complete process and file structure for the old op lowering can be found in [the op lowering guide](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md). Replacing the supported op with the codegen SHOULD NOT introduce any new behavior, it is purely for the clean up purpose. 

## Before you start
You should follow the instructions in [here](https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md) to install required dependencies and build pytorch and pytorch/XLA from the source. You do not need access to TPU to implement the lowering. It is recommended to experiment on a workstation and configure it to use XLA:CPU. You can configure Pytorch/XLA to use XLA:CPU by running

```
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0" XRT_WORKERS="localservice:0;grpc://localhost:51011"
```

It is also recommended that you're familiar with our [op lowering process](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md) before you work on the codegen. 

PyTorch/XLA uses https://github.com/pytorch/xla/issues/3560 to track the status of codegen migration. When working on a codegen, please put your GitHub alias with the PR link on the issue to avoid duplicate work. 

## File structure
All file mentioned below lives under the `xla/torch_xla/csrc` folder, with the exception of `xla_native_functions.yaml`

### PyTorch Codegen files
- torch/csrc/lazy/core/shape_inference.h
  - Shape inference functions defined for each op that will take for input torch::lazy::shapes and return output torch::lazy::shape. Only the ops that is not structural will require a manual shape inference function
- torchgen/gen_lazy_tensor.py
  - Builds on existing data models and helpers used by all ATen backends, and adds new functionality specific to lazy  tensor backends.  run_gen_lazy_tensor is defined in this file
- torchgen/dest/lazy_ir.py
  - Contains data class GenLazyIR that can be overridden by the back and defined the generated IR class

### PyTorch/XLA Codegen files
- xla/xla_native_functions.yaml
  - Contains all the op XLA supported today. Most of the ops are under the supported category, the goal of this document is to move most of the ops to the full_codegen category.
- xla/scripts/gen_lazy_tensor.py
  - Provides necessary XLA versions of the codegen Codegen class and calls the upstream codegen API.
- xla/torch_xla/csrc/XLANativeFunctions.cpp
  - Result of the full_codegen column of the xla/xla_native_functions.yaml. The op function defined here will implement the op declared in the XLANativeFunctions.h. Each op will take at::tensor and return another at::tensor wrapped around a XLATensor.
- xla/torch_xla/csrc/LazyIr.h
  - Result of the full_codegen column of the xla/xla_native_functions.yaml.  Defines the IR that is used to construct the full_codegen ops.

### PyTorch/XLA Old Op Lowering files
- xla/torch_xla/csrc/generated/aten_xla_type.cpp
  - Manually implements ops defined in xla/xla_native_functions.yaml. Will be replaced by XLANativeFunctions.cpp
- xla/torch_xla/csrc/generated/tensor.h
  - Defines XLATensor class and XLATensor method declarations. These declarations are usually a one to one mapping of the at::Tensor nodes we declared in XLANativeFunctions.h. XLATensor method will be removed for full_codegen ops
- xla/torch_xla/csrc/generated/tensor_method.cpp
  - Implements tensor methods defined in tensor.h. This file will be removed for full_codegen ops
- xla/torch_xla/csrc/generated/ops/…
  - Defines IR class for “most” ops. It is possible that multiple ops share the same IR.

## Codegen step by step
### 1. Identify the op 
When you work on your first few codegens, we generally recommend you to start with the simpler ops. This guide will go over one unary one one binary op as examples, but it is recommend that you avoid ops with the following characteristics:
1. Contains custom fallback code. For example in _adaptive_avg_pool3d, there is a conditional fallback:
```
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()),
                               output_size_list, /*pool_dim=*/3)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(_adaptive_avg_pool3d)>::call(self, output_size);
  }
```
2. Results in dynamic shape as these ops are WIP and may evolve over time. At some future point, we may bring the ops into codegen. 
3. Does not invoke a tensor_method directly. For example _copy_from:
```
 if (!self_tensor) {
   static bool sync_update =
       xla::sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true);
   XLA_CHECK(dst_tensor);
   dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
 }
```
4. Has a complicated tensor_method, ideally it should be a directly mapping from op to IR.

An good example of a "simple" op would be something like `abs`:
```
at::Tensor XLANativeFunctions::abs(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::abs(bridge::GetXlaTensor(self)));
}
```

### 2. Codegen the op and inspect the generated file
Find the op in  `xla/xla_native_functions.yaml` and move it to the full_codegen column and run `python setup.py install` under xla directory again. The build will fail (reason explained later in this guide) but you can still see the generated file. The code snippets below uses `abs` as an example.
#### XLANativeFunctions.cpp
```
at::Tensor XLANativeFunctions::abs(const at::Tensor & self) {
  TORCH_LAZY_FN_COUNTER("xla::");
  auto common_device = torch_xla::bridge::GetXlaDevice(self);
  TORCH_INTERNAL_ASSERT(common_device);

  torch_xla::XLATensorPtr lazy_self = torch_xla::bridge::GetXlaTensorOrCreateForWrappedNumber(self, *common_device);

  torch::lazy::NodePtr node = torch::lazy::ReuseNode<Abs>(lazy_self->GetIrValue());
  if (!node) {
    node = torch::lazy::MakeNode<Abs>(lazy_self->GetIrValue());
    CacheNode(node);
  }

  auto result = torch_xla::bridge::AtenFromXlaTensor(
        torch_xla::XLATensor::Create(std::move(node), *common_device));
  return result;
};
```
Describing the generated code line by line:
- Get and verify device from input tensor
```
  auto common_device = torch_xla::bridge::GetXlaDevice(self);
  TORCH_INTERNAL_ASSERT(common_device);
```
- Check if we can reuse the node from previous creation. If not, create corresponding IR node and cache it. 
```
  torch::lazy::NodePtr node = torch::lazy::ReuseNode<Abs>(lazy_self->GetIrValue());
  if (!node) {
    node = torch::lazy::MakeNode<Abs>(lazy_self->GetIrValue());
    CacheNode(node);
  }
```
- Wrap the newly created IR node in a XLATensor. And wrap the XLATensor within the at::Tensor and return it as a result. Note that this part used to be manually done in tensor_method.cpp.
```
  auto result = torch_xla::bridge::AtenFromXlaTensor(
        torch_xla::XLATensor::Create(std::move(node), *common_device));
  return result;
```

#### LazyIr.h
```
class Abs : public XlaNode {
 public:
  Abs(const torch_xla::XlaValue& self)
      : XlaNode(torch::lazy::OpKind(at::aten::abs), {self},
                [&]() { return AbsOutputShape(self); },
                /* num_outputs */ 1, torch::lazy::MHash())
  {}

  std::string ToString() const override {
    std::stringstream ss;
    ss << XlaNode::ToString();
    return ss.str();
  }
  torch_xla::XlaOpVector Lower(LoweringContext* loctx) const override;
};
```

A couple of things to keep in mind:
- Codegen does not generate the `Clone` method which is expected. There is no use of the `Clone` method even in PyTorch/XLA today, we will remove them as part of the migration.
- For every op, it will generate a {OP}OutputShape method. We need to manually declare and implement this method in a separate file.
- For every op, it will generate a Lower declaration. We need to manually implement this lowering function in a separate file.

### 3. Implement the missing IR function
#### torch_xla/csrc/ops/ops_xla_shape_fn.h 
Declare the {OP}OutputShape:
```
xla::Shape AbsOutputShape(const XlaValue& input);
```
#### torch_xla/csrc/ops/ops_xla_shape_fn.cpp
Implement the {OP}OutputShape:
```
xla::Shape AbsOutputShape(const XlaValue& input) { return input.xla_shape(); }
```

`Abs` is an overly simplified example, in a normal case you need to call the BuildXXXOp function again to get the output shape. A slightly better example would be:
```
xla::Shape MaximumOutputShape(const XlaValue& input, const XlaValue& other) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto promoted = XlaHelpers::Promote(operands[0], operands[1]);
    return xla::Max(promoted.first, promoted.second);
  };
  return InferOutputShape({input.xla_shape(), other.xla_shape()},
                          lower_for_shape_fn);
}
```

Note that you should not start from scratch. Find the Xla::Shape computation logic from the existing op and move it this these two files.

### 4. Implement the lowering function
#### torch_xla/csrc/ops/ops_lower_fn.cpp
```
torch_xla::XlaOpVector Abs::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildAbs(xla_input), loctx);
}
```
Note that this function should be directly moved from the existing lowering. Some Ops that were originally implemented in `torch_xla/csrc/ops/ops.cpp` use `GenericOp`. You will need to slightly modify their lowering implementation to fit the implementation provided above.

### 5. Cleanup
Delete the existing op from aten_xla_type.cpp, tensor_methods.h, tensor_methods.cpp, and ops/…. Note that sometimes you have to keep the tensor_method, because it is being used in tensor_ops like. So, before removing the op, cross reference it with `tensor_ops.cpp`.
```
  XLATensor s1 = XLATensor::sub(XLATensor::mul(u2, v3), XLATensor::mul(u3, v2), one);
```
Sometimes other IRNode uses the 'IRNode' you migrated. In this case you need to update those IRNode lowering logic as well. In the long term we need to get rid  of these composite IR from our end and provide a lowering function for each op.
```
  torch::lazy::NodePtr exp = Pow(Abs(input), norm_exp);
```
to
```
  torch::lazy::NodePtr exp =
      Pow(torch::lazy::MakeNode<Abs>(input, std::vector<torch::lazy::Shape>()),
          norm_exp);
```

## Run the test and verify the result
Run the C++ op test or a simple test that only involves the generated ops. To run the C++ test:
1. Build the xla through `python setup.py install` (note: don't use the `BUILD_CPP_TESTS=0` flag since this will skip building the C++ tests)
2. Go into the `test/cpp/build` directory in your `pytorch/xla`
3. Run the command to run the desired C++ test (for example, to run `Abs` C++ test):
```
./test_ptxla --gtest_filter=AtenXlaTensorTest.TestAbs
```
As usual, two things to verify are the correctness and the xla counter being incremented correctly.

## Sample PRs
- Unary/Binary OP -> Codegen erf, erfc, erfinv, and exp (https://github.com/pytorch/xla/pull/3659)
- OP with optional -> Codegen binary_cross_entropy/backward (https://github.com/pytorch/xla/pull/3809)
- OP with `at::Scalar` -> Codegen addcdiv and addcmul (https://github.com/pytorch/xla/pull/3768)
- OP with vector that support negative index -> Codegen amin amax (https://github.com/pytorch/xla/pull/3771)
- OP with special fallback logic -> partially codegen adaptive_avgpool3d and backward (https://github.com/pytorch/xla/pull/3790)
To see more examples, please take a look at the tracking issue (https://github.com/pytorch/xla/issues/3560).
