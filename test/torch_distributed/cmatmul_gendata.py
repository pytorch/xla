import torch
import torch_xla
from torch import nn
import torch.nn.init as init

# intermediate_size = 28672
# hidden_size = 8192
# token_size = 1024

intermediate_size = 28672
hidden_size = 8192
token_size = 1024

torch.manual_seed(123)

class FFN(nn.Module):

  def __init__(self, hidden_dim, intermediate_dim, dtype=torch.bfloat16):
    super().__init__()
    self.fc1 = nn.Linear(hidden_dim, intermediate_dim, dtype=dtype, bias=False)
    self.fc2 = nn.Linear(intermediate_dim, hidden_dim, dtype=dtype, bias=False)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x

def init_weights(m):
  if isinstance(m, nn.Linear):
    init.xavier_uniform_(m.weight)

input = torch.randn((token_size, hidden_size), dtype=torch.bfloat16)
m = FFN(hidden_size, intermediate_size)
m.apply(init_weights)
torch.save(m.state_dict(), "ffn_state_dict.pt")
torch.save(input, "ffn_input.pt")

with torch.no_grad():
    cpu_out = m(input)

input = input.to('xla')
m = m.to('xla')

with torch.no_grad():
    xla_out = m(input)
xla_out_cpu = xla_out.cpu()
# breakpoint()
torch.save(cpu_out, "ffn_cpu_out.pt")
torch.save(xla_out_cpu, "ffn_xla_out.pt")

