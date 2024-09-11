import os
import re
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import unittest

device = xm.xla_device()


class TestAutocastXla(unittest.TestCase):
    def test_cross_entropy_loss(self):
        data = torch.randn(16, 10).to(torch.bfloat16).to(device)
        target = torch.randn(16, 10).to(torch.bfloat16).to(device)
        with torch.autocast("xla"):
            loss = torch.nn.CrossEntropyLoss()(data, target)
            hlo = torch_xla._XLAC._get_xla_tensors_hlo([loss])
            self.assertTrue(
                re.search(rf".*convert.*f32.*convert.*bf16", hlo) is not None
            )

            self.assertTrue(
                re.search(rf".*exponential.*f32.*exponential.*f32", hlo) is not None
            )

            self.assertTrue(re.search(rf".*log.*f32.*log.*f32", hlo) is not None)


if __name__ == "__main__":
    unittest.main()
