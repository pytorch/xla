
import math
import torch

def derive_int8_quant_parameters(w):
    '''
    Create this function for testing purpose, there should be better weight
    quantization algorithms in PyTorch.

    A native function to derive the scaler for int8 weight quantization.
      fp_val = int8_val * 2**(scaler)
    1. Determine the range of the weights
    2. Search the mininum exp (min_exp) such that 2^min_exp * container_range
       covers the range of the weights.
    '''
    weight_range = w.max() - w.min()
    # Find the scaler for the quantized weight. w_float = w_int8 * scaler
    # scaler range: 2^-15, 2^15
    min_exp = -15
    max_exp = 15
    exp = min_exp
    container_range = 2**8
    while exp <= max_exp:
      # 256 is the range of int8
      if math.pow(2, exp) * container_range > weight_range:
          return 2**exp
      exp += 1
    return 2**exp

def quant_weight(w):
    '''
    quant weight 'w' to a int8 tensor
    w: float32 weight
    '''
    container_min = -128
    container_max = 127
    scaler = derive_int8_quant_parameters(w)
    quantized_tensor = w.clone()
    quantized_tensor.detach().apply_(lambda x: round(x / scaler))
    torch.clamp(quantized_tensor, container_min, container_max)
    quantized_tensor = quantized_tensor.to(torch.int8)
    return quantized_tensor, scaler

class LinearQuant(torch.nn.Module):
    '''
    Int8 weight-only quantized linaer
    '''
    def __init__(self, in_feature, out_feature, bias=False):
        super().__init__()
        # Set requires_grad is necessary as tensor with grad doesn't support integer tensors.
        self.int8_weights = torch.nn.Parameter(torch.randint(-128, 127, (in_feature, out_feature), dtype=torch.int8), requires_grad=False)
        # self.scaler = torch.nn.Parameter(torch.rand(1), requires_grad=False)
        self.scaler = torch.rand(1, dtype=torch.bfloat16)
        self.bias = bias
        if bias:
            self.int8_bias = torch.nn.Parameter(torch.rand((out_feature), dtype=torch.int8), requires_grad=False)
            self.bias_scaler = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)

    def forward(self, x):
        fp_weights = self.int8_weights * self.scaler
        x = torch.matmul(x, fp_weights)
        if self.bias:
            fp_bias = self.int8_bias * self.bias_scaler
            x += fp_bias
        return x

    def load_fp_params(self, linear: torch.nn.Linear):
        int8_w, scaler = quant_weight(linear.weight)
        self.int8_weights.copy_(int8_w.transpose(1,0))
        self.scaler.copy_(scaler)
        # TODO add support for bias
