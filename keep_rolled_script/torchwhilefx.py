import torch
from typing import List

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# Reset since we are using a different backend.
torch._dynamo.reset()

def f(x):
    i = 1
    while i < 10:
        x = x + 1
        i = i + 1
    return x

opt_f = torch.compile(f, backend=custom_backend)
inp1 = torch.ones(1, requires_grad=True)
print(opt_f(inp1))
