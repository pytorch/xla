import os
import tempfile
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import torch_xla2

from torch_xla2 import tf_integration
from . import test_base


class Interpolate(torch.nn.Module):

  def forward(self, masks: torch.Tensor) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        size=(500, 500),
        mode="bilinear",
        align_corners=False,
    )
    return masks


class TfIntegrationTest(test_base.TestCase):

  def setUp(self):
    torch.manual_seed(0)
    torch_xla2.enable_accuracy_mode()

  def test_interpolate(self):
    """Simple model roundtripped through TF savedmodel"""

    # Create model
    arg = (torch.randn(3, 3, 200, 200),)
    pt_model = Interpolate()

    # Export to SavedModel
    with tempfile.TemporaryDirectory() as tempdir:
      sm_path = os.path.join(tempdir, "interpolate.savedmodel")
      tf_integration.save_torch_module_as_tf_saved_model(pt_model, arg, sm_path)

      # Reload SM and compare results with PT results
      loaded_model = tf.saved_model.load(sm_path)
      pt_res = pt_model(*arg)
      tf_res = torch.tensor(loaded_model.f(*arg)[0].numpy())
      self.assertTrue(torch.allclose(pt_res, tf_res, atol=1e-4))



if __name__ == "__main__":
  test_base.main()
