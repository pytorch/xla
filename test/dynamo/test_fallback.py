import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch._dynamo as dynamo
import torch._dynamo.test_case as dt


class XlaDynamoFallbackTest(dt.TestCase):

  def test_operator_fallback(self):

    @dynamo.optimize("torchxla_trace_once")
    def fn_fallback(M, mat1, mat2):
      A = torch.cummin(M, 1)
      B = torch.add(A[0], mat1)
      return torch.add(B, mat2)

    M = torch.randn(5, 10, device=xm.xla_device())
    mat1 = torch.randn(5, 10, device=xm.xla_device())
    mat2 = torch.randn(5, 10, device=xm.xla_device())

    res = fn_fallback(M, mat1, mat2)

  def test_operator_operand_fallback(self):

    @dynamo.optimize("torchxla_trace_once")
    def fn_fallback(M, mat1, mat2, beta):
      # xla currently only support alpha and beta == 1
      A = torch.addmm(M, mat1, mat2, beta=beta)
      B = torch.add(A, mat1)
      return torch.add(B, mat1)

    M = torch.randn(2, 3, device=xm.xla_device())
    mat1 = torch.randn(2, 3, device=xm.xla_device())
    mat2 = torch.randn(3, 3, device=xm.xla_device())

    res = fn_fallback(M, mat1, mat2, 0.5)


if __name__ == '__main__':
  from torch._dynamo.test_case import run_tests

  run_tests()
