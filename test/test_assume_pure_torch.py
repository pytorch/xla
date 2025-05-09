from absl.testing import absltest

import torch
import torch_xla
from torch_xla.experimental.assume_pure import assume_pure_torch

class TestAssumePure(absltest.TestCase):
  def test_assume_pure_basic(self):
    # Arrange
    @assume_pure_torch(use_cache=True)
    def simple_torch_function(a, b):
      result = torch.sin(a @ b)
      return result
    # Act
    a = torch.ones((3, 3), device='xla', requires_grad=True)
    actual = simple_torch_function(a, a)
    torch_xla.sync()

    # Assert
    expected = torch.sin(torch.ones(3, 3) @ torch.ones(3, 3))
    torch.testing.assert_close(actual, expected, check_device=False)

# TODO: Support assume_pure_torch with arbitraty inputs.
# class TracingBenchmark(absltest.TestCase):

#   def test_trace_transformer_with_spda_attention(self):
#     num_iterations = 3
#     print(f"\nRunning benchmark with {num_iterations} iterations")

#     import sys
#     import os
#     example_folder = os.path.dirname(os.path.dirname(__file__)) + "/examples"
#     sys.path.append(example_folder)
#     from decoder_only_model import DecoderOnlyConfig, DecoderOnlyModel  # type:ignore

#     config = DecoderOnlyConfig(
#         hidden_size=128,
#         num_hidden_layers=2,
#         intermediate_size=8 * 128,
#         vocab_size=256)
#     model = DecoderOnlyModel(config=config).to('xla')
#     batch_size = 2
#     sequence_length = 8

#     # Generate random input_ids within the range of the vocabulary size
#     input_ids = torch.randint(0, config.vocab_size,
#                               (batch_size, sequence_length)).to('xla')

#     pure_model = deepcopy(model)
#     torch_xla.sync()

#     # Test tracing the model normally.
#     model(input_ids)  # Warm up
#     start_time = time.time()
#     for _ in range(num_iterations):
#       model(input_ids)
#     end_time = time.time()
#     model_time = (end_time - start_time) / num_iterations
#     print(f"No `@assume_pure` time: {model_time * 1000:.4f} ms")

#     # Test tracing the model with assume_pure.
#     @assume_pure_torch(use_cache=True)
#     def pure_call(params, x):
#       return torch.func.functional_call(pure_model, params, x)

#     params = dict(pure_model.named_parameters())
#     pure_call(params, input_ids)  # Warm up
#     start_time = time.time()
#     for _ in range(num_iterations):
#       pure_call(params, input_ids)
#     end_time = time.time()
#     pure_model_time = (end_time - start_time) / num_iterations
#     print(f"`@assume_pure` time: {pure_model_time * 1000:.4f} ms")

if __name__ == "__main__":
  torch.set_default_dtype(torch.float32)
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  absltest.main()