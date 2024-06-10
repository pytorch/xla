Quantized Operations for XLA device (Experimental feature)
--------------------------

This document outlines how to utilize quantized operations to enable quantization on XLA devices.

XLA Quantized ops offer a high-level abstraction for quantized operations (e.g., blockwise int4 quantized matrix multiplication). These ops are analogous to quantized CUDA kernels ([example](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/gptq/q_gemm.cu)) in the CUDA ecosystem, providing similar functionality and performance benefits within the XLA framework.

**NOTE:** Currently this is classified as experimental feature. It's API specifics 
will change in the next (2.5) release.


## How to use:

XLA quantized operations can be used as `torch op`, or a `torch.nn.Module` that wraps the `torch.op`. These 2 options give model developers the flexibility to choose the best way to integrate XLA quantized ops into their solution.

Both `torch op` and `nn.Module` are compatible with `torch.compile( backend='openxla')`.

### Call XLA quantized op in model code

Users can call XLA quantized ops in the same way as calling other regular PyTorch ops. This provides maximum flexibility in integrating XLA quantized ops into their applications. The quantized ops work in both eager mode and Dynamo, with regular PyTorch CPU tensor and XLA tensor.

**Note** Please check the docstring of the quantized ops for the layout of the quantized weights.

```Python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_quantized_matmul

N_INPUT_FEATURES=10
N_OUTPUT_FEATURES=20
x = torch.randn((3, N_INPUT_FEATURES), dtype=torch.bfloat16)
w_int = torch.randint(-128, 127, (N_OUTPUT_FEATURES, N_INPUT_FEATURES), dtype=torch.int8)
scaler = torch.randn((N_OUTPUT_FEATURES,), dtype=torch.bfloat16)

# Call with torch CPU tensor (For debugging purpose)
matmul_output = torch.ops.xla.quantized_matmul(x, w_int, scaler)

device = xm.xla_device()
x_xla = x.to(device)
w_int_xla = w_int.to(device)
scaler_xla = scaler.to(device)

# Call with XLA Tensor to run on XLA device
matmul_output_xla = torch.ops.xla.quantized_matmul(x_xla, w_int_xla, scaler_xla)

# Use with torch.compile(backend='openxla')
def f(x, w, s):
  return torch.ops.xla.quantized_matmul(x, w, s)

f_dynamo = torch.compile(f, backend="openxla")
dynamo_out_xla = f_dynamo(x_xla, w_int_xla, scaler_xla)
```

It's common to wrap the quantized op into a custom `nn.Module` in model developers model code:

```Python
class MyQLinearForXLABackend(torch.nn.Module):
  def __init__(self):
    self.weight = ...
    self.scaler = ...

  def load_weight(self, w, scaler):
    # Load quantized Linear weights
    # Customized way to preprocess the weights
    ...
    self.weight = processed_w
    self.scaler = processed_scaler

  
  def forward(self, x):
    # Do some random stuff with x
    ...
    matmul_output = torch.ops.xla.quantized_matmul(x, self.weight, self.scaler)
    # Do some random stuff with matmul_output
    ...
```

### Module Swap

Alternatively, users can also use the `nn.Module` that wraps the XLA quantized ops and do module swap in the model code:

```Python
orig_model = MyModel()
# Quantize the model and get quantized weights
q_weights = quantize(orig_model)
# Process the quantized weight to the format that XLA quantized op expects.
q_weights_for_xla = process_for_xla(q_weights)

# Do module swap
q_linear = XlaQuantizedLinear(self.linear.in_features,
                              self.linear.out_features)
q_linear.load_quantized_weight(q_weights_for_xla)
orig_model.linear = q_linear
```

## Supported Quantized Operations:

### Matrix Multiply

| Weight Quantization Type | Activation Quantization Type | Dtype | Supported |
|---|---|---|---|
| per-channel | N/A | W8A16 | Yes |
| per-channel | N/A | W4A16 | Yes |
| per-channel | per-token | W8A8 | No |
| per-channel | per-token | W4A8 | No |
| blockwise | N/A | W8A16 | No |
| blockwise | N/A | W4A16 | No |
| blockwise | per-token | W8A8 | No |
| blockwise | per-token | W4A8 | No |

**Note** `W[X]A[Y]` refers to Weight in `X`-bit, Activation in `Y`-bit. If `X/Y` is 4 or 8, it refers to `int4/8`. 16 for `bfloat16` format.

### Embedding
To be added
