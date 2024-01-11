import torch
from functorch.experimental import control_flow
from torch_xla import stablehlo
from typing import List

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# Reset since we are using a different backend.
torch._dynamo.reset()


def f(i, x):
  return x + i

def map1(x):
  return control_flow.map(f, torch.ones(10), x)

opt_map1 = torch.compile(map1, backend=custom_backend)


args = torch.ones(1, requires_grad=True) # (torch.randn((10, )), )
result = opt_map1(args)

print("args", args)
print("result", result)
