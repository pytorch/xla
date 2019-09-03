import torch
import torch_xla
import torch.nn as nn
import pdb

device = 'xla:0'
# device='cpu'
#def _pad_circular(input, padding):
#    # input = torch.cat([input, input[:, :, 0:padding[-1]]], dim=2)
#    print(input)
#    # input = torch.cat([input[:, :, -(padding[-1] + padding[-2]):-padding[-1]], input], dim=2)
#    # out = input[:, :, -(padding[-1] + padding[-2]):-padding[-1]]
#    print(-(padding[-1] + padding[-2]))
#    print(-padding[-1])
#    out = input[:, :, -1:0]
#    return out
#
#padding = (1, 0)
#x = torch.rand(2, 3, 5, device=device)
#y = _pad_circular(x, padding)
#print(y.cpu())

m_cpu = nn.Conv1d(3, 4, 2, 2, (1,), 1, 1, True, 'circular')
m = nn.Conv1d(3, 4, 2, 2, (1,), 1, 1, True, 'circular').to(device)

input_cpu = torch.rand(2, 3, 5, requires_grad=True)
input = input_cpu.to(device).detach()
input.requires_grad = True

output_cpu = m_cpu(input_cpu)
output = m(input)
pdb.set_trace()

print(output_cpu.shape)
print(output.shape)
