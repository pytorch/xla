# How torch dispatch works

References:
* [__torch_dispatch__](https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557)
* [Dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) Note: old but not outdated.

## torch ops vs. Aten ops

torch ops - regular python functions / methods -> `__torch_functions__`
aten ops - pybind11 registered C++ functions + `OpOverload` wrapper -> `__torch_dispatch__`


## Issues with torch functions:

* More torch functions than *core* Aten ops
* Some torch functions are NOT overridable.

## Issues with torch dispatch:

* Undesired Decompositions
* overloads

## Ways to override

* Subclass
* Decorator

## extension poster
* https://docs.google.com/presentation/d/1piuv9nBzyoqdH49D1SoE5OZUPSMpOOFqfSKOhr-ab2c/edit#slide=id.p1

## How does it works

**TODO:** replace with github links

https://source.corp.google.com/piper///depot/google3/third_party/py/torch/_tensor.py;l=439?q=class%20Tensor&ss=piper%2FGoogle%2FPiper:google3%2Fthird_party%2Fpy%2Ftorch%2F

https://source.corp.google.com/piper///depot/google3/third_party/py/torch/torch/csrc/utils/python_arg_parser.cpp;l=394?q=%22%5C%22__torch_dispatch__%5C%22%22&ss=piper%2FGoogle%2FPiper:google3%2Fthird_party%2Fpy%2Ftorch%2F

https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
