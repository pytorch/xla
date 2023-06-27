import torch
import torch_xla
import torch_xla.core.xla_model as xm



# Modified argmin to call C++ code
# C++ argmin only accepts: vectors of 4 elements with a float dtype
def test_custom_sum():
    t = torch.tensor([6, 2, -1, 23], dtype=torch.float)
    expected = t.argmin()
    print("Correct result", expected)

    x = torch.randn(2048, dtype=torch.float, device=xm.xla_device())
    result = x.sum()
    print("XLA result", result)


if __name__ == '__main__':
    test_custom_sum()