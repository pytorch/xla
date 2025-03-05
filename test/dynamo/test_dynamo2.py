import unittest
from torch_xla._dynamo.dynamo_backend2 import dynamo_backend


class DynamoBackend2Test(unittest.TestCase):
  
  def simple_test(self):
    
    class M(torch.nn.Module):
      
      def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10)

      def forward(self, x):
        return self.a(x)

    dev = torch_xla.device() 
    m = M().to(dev)
    sample_input = torch.randn(10, device=dev)
    m_compiled = torch.compile(m, backend=dynamo_backend2)
    print(m_compiled(sample_input))


if __name__ == '__main__':
  unittest.main()
