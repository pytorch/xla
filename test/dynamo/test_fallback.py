import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch._dynamo as dynamo
import torch._dynamo.test_case as dt


class XlaDynamoFallbackTest(dt.TestCase):

  def test_operator_fallback(self):

    def fn_fallback(M, mat1, mat2):
      A = torch.cummin(M, 1)
      B = torch.add(A[0], mat1)
      return torch.add(B, mat2)

    dynamo_fn = torch.compile(fn_fallback, backend="torchxla_trace_once")
    M = torch.randn(5, 10)
    mat1 = torch.randn(5, 10)
    mat2 = torch.randn(5, 10)
    xla_M = M.to(xm.xla_device())
    xla_mat1 = mat1.to(xm.xla_device())
    xla_mat2 = mat2.to(xm.xla_device())

    cpu_res = fn_fallback(M, mat1, mat2)
    xla_res = dynamo_fn(xla_M, xla_mat1, xla_mat2)

    self.assertTrue(torch.allclose(cpu_res, xla_res.cpu()))

  def test_operator_operand_fallback(self):

    def fn_fallback(M, mat1, mat2, beta):
      # xla currently only support alpha and beta == 1
      A = torch.addmm(M, mat1, mat2, beta=beta)
      B = torch.add(A, mat1)
      return torch.add(B, mat1)

    dynamo_fn = torch.compile(fn_fallback, backend="torchxla_trace_once")
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    xla_M = M.to(xm.xla_device())
    xla_mat1 = mat1.to(xm.xla_device())
    xla_mat2 = mat2.to(xm.xla_device())

    cpu_res = fn_fallback(M, mat1, mat2, 0.5)
    xla_res = dynamo_fn(M, mat1, mat2, 0.5)

    self.assertTrue(torch.allclose(cpu_res, xla_res.cpu()))

  def test_fallback_two_submodules(self):

    def fn_fallback(M, mat1, mat2, beta):
      A = torch.add(M, mat1)
      B = torch.add(A, mat1)
      # xla currently only support alpha and beta == 1
      C = torch.addmm(B, mat1, mat2, beta=beta)
      D = torch.add(C, mat1)
      return torch.add(D, mat1)

    dynamo_fn = torch.compile(fn_fallback, backend="torchxla_trace_once")
    M = torch.randn(2, 3)
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    xla_M = M.to(xm.xla_device())
    xla_mat1 = mat1.to(xm.xla_device())
    xla_mat2 = mat2.to(xm.xla_device())

    cpu_res = fn_fallback(M, mat1, mat2, 0.5)
    xla_res = dynamo_fn(M, mat1, mat2, 0.5)

    self.assertTrue(torch.allclose(cpu_res, xla_res.cpu()))


if __name__ == '__main__':
  from torch._dynamo.test_case import run_tests

  run_tests()
