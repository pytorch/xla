import torch
import torch_xla.core.xla_model as xm
import unittest


class TestAmp(unittest.TestCase):

  def test_amp_update_scale(self):
    device = xm.xla_device()
    growth_tracker = torch.tensor(0, dtype=torch.int32, device=device)
    current_scale = torch.tensor(4, dtype=torch.float, device=device)
    found_inf = torch.tensor(0, dtype=torch.float, device=device)
    scale_growth_factor = 2.0
    scale_backoff_factor = 0.5
    growth_interval = 3
    current_scale = torch._amp_update_scale(growth_tracker, current_scale,
                                            found_inf, scale_growth_factor,
                                            scale_backoff_factor,
                                            growth_interval)
    self.assertAlmostEqual(current_scale.item(), 4.0)
    self.assertEqual(growth_tracker.item(), 1)
    current_scale = torch._amp_update_scale(growth_tracker, current_scale,
                                            found_inf, scale_growth_factor,
                                            scale_backoff_factor,
                                            growth_interval)
    self.assertAlmostEqual(current_scale.item(), 4.0)
    self.assertEqual(growth_tracker.item(), 2)
    current_scale = torch._amp_update_scale(growth_tracker, current_scale,
                                            found_inf, scale_growth_factor,
                                            scale_backoff_factor,
                                            growth_interval)
    self.assertAlmostEqual(current_scale.item(), 8.0)
    self.assertEqual(growth_tracker.item(), 0)
    found_inf = torch.tensor(1, dtype=torch.float, device=device)
    current_scale = torch._amp_update_scale(growth_tracker, current_scale,
                                            found_inf, scale_growth_factor,
                                            scale_backoff_factor,
                                            growth_interval)
    self.assertAlmostEqual(current_scale.item(), 4.0)
    self.assertEqual(growth_tracker.item(), 0)

  def test_amp_foreach_non_finite_check_and_unscale(self):
    device = xm.xla_device()
    grads = [torch.tensor([1, 2, 3, 4], dtype=torch.float, device=device)]
    inv_scale = torch.tensor(0.2, dtype=torch.float, device=device)
    found_inf = torch.tensor(0, dtype=torch.float, device=device)
    torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf,
                                                     inv_scale)
    self.assertAlmostEqual(found_inf.item(), 0.0)

    grads = [
        torch.tensor([1, 2, 3, float('nan')], dtype=torch.float, device=device),
        torch.tensor([1, 2, 3, 5], dtype=torch.float, device=device)
    ]
    torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf,
                                                     inv_scale)
    self.assertAlmostEqual(found_inf.item(), 1.0)


if __name__ == '__main__':
  unittest.main()
