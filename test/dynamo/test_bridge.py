import copy

import torch
import torch_xla

import torch._dynamo.test_case
import torch._dynamo.testing
from functorch.compile import aot_module_simplified, make_boxed_compiler
from torch._dynamo import disable

import torch_xla._dynamo.dynamo_bridge as bridge
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as metrics
from torch import fx, nn


class BasicModule(nn.Module):

  def __init__(self):
    super(BasicModule, self).__init__()

  def forward(self, x, y):
    return x + y

  def get_random_inputs(self):
    return (torch.randn(10), torch.randn(10))


class MatmulModule(nn.Module):

  def __init__(self):
    super(MatmulModule, self).__init__()

  def forward(self, x, y):
    return x @ y

  def get_random_inputs(self):
    return (torch.randn(5, 100), torch.randn(100, 5))


class LinearModule(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(10, 5)

  def forward(self, x):
    return self.linear(x)

  def get_random_inputs(self):
    return (torch.randn(2, 10),)


class MaxPoolModule(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=2)
    self.pool = nn.MaxPool2d(3, stride=2)

  def forward(self, x):
    x = self.conv(x)
    return self.pool(x)

  def get_random_inputs(self):
    return (torch.randn(2, 3, 10, 10),)


class ModuleInplaceUpdate(nn.Module):

  def __init__(self):
    super(ModuleInplaceUpdate, self).__init__()

  def forward(self, a, b):
    a.sub_(b)
    return b - 1, b + 1

  def get_random_inputs(self):
    return (torch.randn(10), torch.randn(10))


class UpsampleModule(nn.Module):

  def __init__(self):
    super().__init__()
    self.upsample = nn.Upsample(scale_factor=2)

  def forward(self, x):
    return self.upsample(x)

  def get_random_inputs(self):
    return (torch.randn((1, 1, 5)),)


def allclose(expected, actual):

  def unwrap(cont):
    if isinstance(cont, (list, tuple)) and len(cont) == 1:
      return cont[0]
    return cont

  expected = unwrap(expected)
  actual = unwrap(actual)

  if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
    return torch.allclose(expected, actual, rtol=1e-03, atol=1e-04)
  elif isinstance(expected,
                  (tuple, list)) and isinstance(actual, (tuple, list)):
    return len(expected) == len(actual) and all(
        torch.allclose(a, b, rtol=1e-03, atol=1e-04)
        for a, b in zip(expected, actual))
  else:
    raise RuntimeError("Unexpected types")


def make_reuse_graph_test(module_class, niter=100):

  def test_wrapper(self):
    xla_dev = xm.xla_device()
    xla_module = module_class().to(device=xla_dev)
    inputs = tuple(x.to(device=xla_dev) for x in xla_module.get_random_inputs())
    metrics.clear_counters()
    optimized_mod = bridge.extract_compiled_graph(
        fx.symbolic_trace(xla_module), inputs)

    for i in range(niter):
      xla_inputs = tuple(
          inp.to(device=xla_dev) for inp in xla_module.get_random_inputs())
      xla_inputs_copy = copy.deepcopy(xla_inputs)

      expected = xla_module(*xla_inputs)
      # make sure above lazy computation is executed.
      torch_xla.sync()

      actual = optimized_mod(*xla_inputs_copy)

      if not allclose(expected, actual):
        print(
            f"Incorrect results at iter {i}. expected\n{expected}, actual\n{actual}"
        )
        self.assertTrue(False)

      # make sure arguments match after calling the model forward method
      # to handle inplace updates.
      if not allclose(xla_inputs, xla_inputs_copy):
        print(
            f"Incorrect updated arguments at iter {i}. expected\n{xla_inputs}, actual\n{xla_inputs_copy}"
        )
        self.assertTrue(False)

  return test_wrapper


def training_compiler(gm, example_inputs):

  @make_boxed_compiler
  @disable
  def fw_compiler(graph, inputs, *args, **kwargs):
    # tracing time inputs are FakeTensors, we can not pass them
    # to extract_compiled_graph directly since we can not extract
    # xla tensor id from fake tensors. Call extract_compiled_graph
    # lazily and trigger that for the first call with non-fake tensors.
    compiled_graph = None

    def optimized_mod(*args):
      nonlocal compiled_graph
      if compiled_graph is None:
        compiled_graph = bridge.extract_compiled_graph(graph, args)
      return compiled_graph(*args)

    return optimized_mod

  return aot_module_simplified(gm, example_inputs, fw_compiler=fw_compiler)


def model_iter_fn_train(mod, inputs):
  outputs = mod(*inputs)
  loss = outputs.mean()
  loss.backward()

  param_list = list(mod.parameters())
  return [param.grad for param in param_list]


def make_training_test(model_cls):

  def test_wrapper(self):
    import torch_xla.core.xla_model as xm

    xla_dev = xm.xla_device()
    model = model_cls()
    inputs = model.get_random_inputs()

    model = model.to(device=xla_dev)
    inputs = tuple(inp.to(device=xla_dev) for inp in inputs)
    inputs = tuple(inp.requires_grad_() for inp in inputs)

    # do baseline
    baseline_model = copy.deepcopy(model)
    baseline_inputs = copy.deepcopy(inputs)
    expected_output = model_iter_fn_train(baseline_model, baseline_inputs)

    compiler = training_compiler
    optimize_ctx = torch._dynamo.optimize(compiler, nopython=False)
    optimized_model_iter_fn = optimize_ctx(model_iter_fn_train)

    actual_output = optimized_model_iter_fn(model, inputs)
    print(
        f"expected_output:\n{expected_output}\nactual_output:\n{actual_output}")
    assert allclose(expected_output, actual_output)

  return test_wrapper


class TorchXLAReuseGraphTest(torch._dynamo.test_case.TestCase):

  test_basic = make_reuse_graph_test(BasicModule)
  test_matmul = make_reuse_graph_test(MatmulModule)
  test_linear = make_reuse_graph_test(LinearModule)
  test_inplace_update = make_reuse_graph_test(ModuleInplaceUpdate)

  test_training_linear = make_training_test(LinearModule)
  test_training_maxpool = make_training_test(MaxPoolModule)
  test_training_upsample = make_training_test(UpsampleModule)

  def _compile_and_check(self, fn, args, backend="openxla"):
    r = fn(*args)
    torch_xla.sync()

    compiled_fn = torch.compile(backend=backend)(fn)
    compiled_r = compiled_fn(*args)
    torch_xla.sync()

    self.assertEqual(r, compiled_r)

  def test_non_tensor_args_for_partition(self):

    class Emb(torch.nn.Embedding):

      def __init__(self):
        super().__init__(num_embeddings=10, embedding_dim=10, padding_idx=0)

    device = xm.xla_device()
    module = Emb()
    module.to(device)

    def foo(x):
      return module(x)

    x = torch.randint(0, 10, (10,), device=device)
    self._compile_and_check(foo, (x,), backend="openxla")

  def test_inputs_not_computed(self):

    def foo(x):
      return x * 2

    device = xm.xla_device()
    x = torch.rand(5, device=device)
    x = x.unsqueeze(dim=-1)
    self._compile_and_check(foo, (x,))

  def test_factory_copy(self):

    def foo(device):
      return torch.arange(5, device="cpu").to(device)

    self._compile_and_check(foo, (xm.xla_device(),))

  def test_index_flag_unsupported(self):
    # The indices of the index operation are represented as
    # a list of objects. If any non-XLA tensors appear, the
    # index operation should be flagged as unsupported, since
    # their arguments might be turned into placeholders of the
    # partition FX graph.

    def foo(xt, t):
      return xt[t]

    device = xm.xla_device()
    xt = torch.rand(5, device=device)
    t = torch.randint(0, 5, (3,))
    self._compile_and_check(foo, (xt, t))

  def test_stack_flag_unsupported(self):
    # Explicit list of tensors arguments.

    def foo(t):
      return torch.stack([t])

    t = torch.randint(0, 5, (3,))
    self._compile_and_check(foo, (t,))

  def test_cpu_flag_unsupported(self):
    # Nodes that return CPU tensors should also be flagged as
    # unsupported, since their outputs could be turned into
    # outputs of the partition FX graph.

    def foo(t):
      return t.cpu()

    device = xm.xla_device()
    t = torch.randint(0, 5, (3,), device=device)
    self._compile_and_check(foo, (t,))


if __name__ == "__main__":
  from torch._dynamo.test_case import run_tests

  run_tests()
