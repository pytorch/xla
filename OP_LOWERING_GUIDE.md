# OP Lowering Guide

## Background
PyTorch wraps the C++ ATen tensor library that offers a wide range of operations implemented on GPU and CPU. Pytorch/XLA is a PyTorch extension; one of its purposes is to convert PyTorch operations to XLA operations. Lowering defines a process of converting a higher-level representation to a lower-level representation. In this document, I will refer to the process of converting PyTorch operation to XLA operation as the lowering. XLA Compiler will also lower XlaOp to HLO, but that’s beyond the scope of this documentation. We will forward operations that we haven’t provided an XLA lowering yet to CPU and call ATen implementations. Operations that are forwarded to the CPU will cause a significant slowdown. We must lower all operations used in the model to achieve the best performance.

## Before you start
You should follow the instructions in [here](https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md) to install required dependencies and build pytorch and pytorch/XLA from the source. You do not need access to TPU to implement the lowering. It is recommended to experiment on a workstation and configure it to use XLA:CPU. You can configure Pytorch/XLA to use XLA:CPU by running

```
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0" XRT_WORKERS="localservice:0;grpc://localhost:51011"
```

## Understanding the operation
You can find the definition of the C++ ATen operations in [native_functions.yaml](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml). After you build Pytorch/XLA from source, you will also find our default implementation (a boxed kernel which forwards calls to PyTorch native CPU) in `xla/torch_xla/csrc/aten_cpu_fallback.h/cpp`. Pytorch operations can usually be mapped to [PyTorch tensor api](https://pytorch.org/docs/stable/index.html) easily. If that is not the case searching the PyTorch native implementation under [PyTorch repo](https://github.com/pytorch/pytorch) is recommended. The goal is to lower the PyTorch operations into a sequence of XLA operations defined in [here](https://www.tensorflow.org/xla/operation_semantics).

## File structure
All file mentioned below lives under the `xla/torch_xla/csrc` folder, with the exception of `codegen/xla_native_functions.yaml`

1. `xla_native_functions.yaml` contains the list of all operators that are lowered. Each operator name must directly match a pytorch operator listed in [native_functions.yaml](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml). This file serves as the interface to adding new xla operators, and is an input to PyTorch's [codegen machinery](https://github.com/pytorch/pytorch/blob/master/torchgen/gen_backend_stubs.py). It generates the below 3 files: `XLANativeFunctions.h`, `RegisterXLA.cpp`, and `RegisterAutogradXLA.cpp`
2. `XLANativeFunctions.h` and `aten_xla_type.cpp` are entry points of PyTorch to the pytorch_xla world, and contain the manually written lowerings to XLA for each operator. `XLANativeFunctions.h` is auto-generated through a combination of `xla_native_functions.yaml` and the PyTorch core `native_functions.yaml` file, and contains declarations for kernels that need to be defined in `aten_xla_type.cpp`. The kernels written here need to construct 'XLATensor' using the input `at::Tensor` and other parameters. The resulting `XLATensor` needs to be converted back to the `at::Tensor` before returning to the PyTorch world.
3. `RegisterXLA.cpp` and `RegisterAutogradXLA.cpp` are auto-generated files that register all lowerings to the PyTorch Dispatcher. They also include auto-generated wrapper implementations of `out=` and `inplace` operators.
4. `aten_cpu_fallback.h/.cpp` contain our boxed fallback implementation to CPU. The boxed fallback kernel will be used if a lowering is not explicitly defined in `xla_native_functions.yaml` + `aten_xla_type.cpp`, and the operator is not composite.
5. `tensor_methods.h` contains the `XLATensor` declarations. These declarations are usually a one to one mapping of the `at::Tensor` nodes we declared in `XLANativeFunctions.h`
6. `tensor_methods.cpp` contains the implementation of `XLATensor node` defined in `tensor_methods.h`. We constructed the corresponding `ir::op` from the parameter’s `ir::Value` and wrapped it inside a `XLATensor`. Ir stands for intermediate representation.
7. `ops/` directory contains all `ir::ops` declaration and definition. Smaller nodes can be put in `ops/ops.h/.cpp`. More complicated nodes can be put into a separate file. All ops inherit from `ir::ops::Node` and provide a way to lower input `ir::Value` to a sequence of `XlaOp`.

## Unit Test
Our CircleCI runs PyTorch native python tests for every change and every day. Those tests will use XLA implementation if we provide a lowering. We usually don’t need to add additional python tests for PyTorch/XLA unless we want to verify some xla behaviors(like dynamic shape) or we skipped the pytorch native test for some reason. The python test should be added to `xla/test/test_operations.py` if it is required. We also need to add CPP tests in `xla/test/cpp/test_aten_xla_tensor.cpp`. This test should call PyTorch c++ API and verify our implementation yields the same result as PyTorch native implementation. We also need to verify if the xla implementation is called when the tensor is a XLA tensor by checking the `aten::op` and `xla::op` counters.

## Tips
The process of lowering is breaking down the PyTorch operations into a sequence of XlaOp. To provide a good lowering of the PyTorch operation, one needs to have a good grasp of what XLA is capable of. Reading the XlaOp document and looking into how similar ops is lowered is the best way to achieve that. You can find a minimal Op lowering example in [this pr](https://github.com/pytorch/xla/pull/2969). You can also find a slightly more complicated example with backward lowering in [this pr](https://github.com/pytorch/xla/pull/2972).

We have auto-generated wrapper implementations of `out=` and `inplace` operators for some operators in `RegisterXLA.cpp`. We only need to lower the vanilla op in this case. An example would be `lerp` operator which has 6 variants in `native_functions.yaml`, they are

```
  - lerp_.Scalar
  - lerp_.Tensor
  - lerp.Scalar_out
  - lerp.Tensor_out
  - lerp.Scalar
  - lerp.Tensor
```

and will generate function prototypes

```
at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight);
at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight);
at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight);
at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out);
at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight);
at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out);
```

in `XLANativeFunctions.h` if we add all of them to the `xla_native_functions.yaml`. However if we only lower `lerp.Scalar` and `lerp.Tensor` and check `RegisterXLA.cpp`, we will see

```
namespace {

at::Tensor wrapper_Scalar_lerp(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
    // No device check


  // DeviceGuard omitted
  return torch_xla::lerp(self, end, weight);
}

} // anonymous namespace

at::Tensor & wrapper_Scalar_lerp_(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  auto wrapper_Scalar_lerp__tmp = wrapper_Scalar_lerp(self, end, weight);
  at::_copy_from(wrapper_Scalar_lerp__tmp, self);
  return self;
}

...
  m.impl("lerp_.Scalar",
  TORCH_FN(wrapper_Scalar_lerp_));

```

The codegen will automatically generate lowerings for `lerp_.Scalar` and `lerp.Scalar_out` that use our `lerp.Scalar` implementation, without us having to provide an explicit lowering.

In general, if there is an operator in pytorch core that has both an out-of-place and an out= variant, it's better to write a lowering for the out-of-place variant, since you'll get a code-generated out= lowering for free.

For each node we need to pass an `ir::OpKind`. Here is an ([example](https://github.com/pytorch/xla/blob/5ce99bff336325feb41a982dc80299fb53166b29/torch_xla/csrc/ops/var_mean.cpp#L36)). You can find the `OpKind` definition in [aten_interned_strings.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/aten_interned_strings.h) or [interned_strings.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/interned_strings.h). If the aten symbol is missing, you can submit a PR like [this](https://github.com/pytorch/pytorch/pull/36851).
