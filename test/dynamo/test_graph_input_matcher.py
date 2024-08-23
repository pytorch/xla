import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch import nn
from torch.utils._pytree import tree_map_only
from torch_xla._dynamo.dynamo_bridge import GraphInputMatcher


class M(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(5, 3)

  def forward(self, x):
    return self.linear(x)

  def get_example_inputs(self):
    return (torch.rand(10, 5),)


class TestGraphInputMatcher(unittest.TestCase):

  def test_no_cache_fx_gragh_inputs(self):
    xla_dev = xm.xla_device()
    model = M().to(device=xla_dev)
    inputs = tree_map_only(torch.Tensor, lambda x: x.to(device=xla_dev),
                           model.get_example_inputs())

    xm.mark_step()
    args_tensor_ids = [
        torch_xla._XLAC._xla_get_tensor_id(xla_arg) for xla_arg in inputs
    ]
    tensor_id_to_arg_idx = {
        tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)
    }
    output = model(*inputs)
    xla_graph_hash = torch_xla._XLAC._get_graph_hash([output])
    (
        graph_input_tensor_ids,
        graph_input_xla_values,
    ) = torch_xla._XLAC._get_tensors_xla_device_data_node([output])
    xla_args_tensor_ids = set(
        tree_map_only(torch.Tensor,
                      lambda input: torch_xla._XLAC._xla_get_tensor_id(input),
                      inputs))
    graph_input_matcher = GraphInputMatcher(tensor_id_to_arg_idx,
                                            graph_input_tensor_ids,
                                            graph_input_xla_values,
                                            xla_args_tensor_ids)
    # The weight and bias are cached in GraphInputMatcher,
    # the model input will not be cached.
    self.assertEqual(graph_input_matcher.graph_input_xla_values.count(None), 1)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
