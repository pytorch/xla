from __future__ import print_function

import sys; sys.path.insert(0, '/usr/share/torch-xla-nightly/pytorch/xla')


import unittest

from torch_xla.debug import metrics_compare_utils as mcu

class MetricsCompareUtilsTest(unittest.TestCase):

  #Accumulator: 07s370ms955.021us
  def test_tempp(self):
    self.assertTrue(mcu._parse_metrics_report("""Metric: TransferFromServerTime
  TotalSamples: 1728
  Accumulator: 10GB
  ValueRate: 024ms258.635us / second
  Rate: 16.8665 / second
  Percentiles: 1%=909.826us; 5%=997.067us; 10%=001ms45.729us; 20%=001ms92.220us; 50%=001ms216.204us; 80%=002ms513.601us; 90%=002ms872.090us; 95%=002ms290.191us; 99%=006ms818.808us
Metric: TransferToServerTime
  TotalSamples: 2616
  Accumulator: 01m29s615ms262.004us
  ValueRate: 783ms426.227us / second
  Rate: 24.5054 / second
  Percentiles: 1%=003ms783.790us; 5%=004ms98.116us; 10%=010ms364.458us; 20%=015ms121.974us; 50%=026ms622.656us; 80%=035ms554.304us; 90%=082ms478.207us; 95%=108ms554.247us; 99%=129ms338.132us
Counter: CachedSyncTensors
  Value: 11336
Counter: CreateCompileHandles
  Value: 40
Counter: CreateDataHandles
  Value: 407992"""))

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
