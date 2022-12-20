import os
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest


class MetricsTest(unittest.TestCase):

  def test_short_metrics_report_default_list(self):
    begin = time.perf_counter_ns()
    xla_device = xm.xla_device()
    t1 = torch.tensor(1456, device=xla_device)
    t2 = t1 * 2
    t2 = t2 ** 2
    t2 = t2 ** 2
    t2 = t2 ** 2
    t2 = t2 ** 2
    xm.mark_step()
    t2_cpu = t2.cpu()
    wall_time_ns = time.perf_counter_ns() - begin
    self.assertIn("ExecuteTime", met.metric_names())
    execute_time_ns = met.metric_data('ExecuteTime')[1]
    self.assertGreater(wall_time_ns, 2*execute_time_ns)
    print(execute_time_ns, wall_time_ns)


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
