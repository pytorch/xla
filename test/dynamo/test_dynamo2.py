import unittest
import torch
import torch_xla
from torch_xla._dynamo.dynamo_backend2 import dynamo_backend


class DynamoBackend2Test(unittest.TestCase):
  
  def test_simple(self):
    
    class M(torch.nn.Module):
      
      def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10)

      def forward(self, x):
        return self.a(x)

    dev = torch_xla.device() 
    m = M().to(dev)
    sample_input = torch.randn(10, device=dev)
    eager_out = m(sample_input)
    m_compiled = torch.compile(m, backend=dynamo_backend)
    compiled_out = m_compiled(sample_input)
    
    torch.testing.assert_close(eager_out.cpu(), compiled_out.cpu())


if __name__ == '__main__':
  unittest.main()
