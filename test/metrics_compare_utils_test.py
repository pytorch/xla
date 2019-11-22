from __future__ import print_function

import sys; sys.path.insert(0, '/usr/share/torch-xla-nightly/pytorch/xla')


import unittest

from torch_xla.debug import metrics_compare_utils as mcu

_REPORT_1 = """Metric: InboundData
  TotalSamples: 1728
  Accumulator: 10GB
  Rate: 16.8665 / second
Metric: TransferToServerTime
  TotalSamples: 2616
  Accumulator: 01m29s615ms
  ValueRate: 783ms426.227us / second
  Rate: 24.5054 / second
  Percentiles: 1%=003ms783.790us; 5%=004ms98.116us; 10%=010ms364.458us; 20%=015ms121.974us; 50%=026ms622.656us; 80%=035ms554.304us; 90%=082ms478.207us; 95%=108ms554.247us; 99%=129ms338.132us
Counter: CachedSyncTensors
  Value: 11336
Counter: CreateCompileHandles
  Value: 40
Counter: CreateDataHandles
  Value: 407992"""

_REPORT_2 = """--metric:localhost\n\n\nMetric: InboundData
  TotalSamples: 73216
  Accumulator: 64.75TB
Metric: TransferToServerTime
  TotalSamples: 247016
  Accumulator: 04d17h11m07s495ms546.299us
Counter: CachedSyncTensors
  Value: 1022168
Counter: CreateCompileHandles
  Value: 40
Counter: CreateDataHandles
  Value: 576462152
"""

_REPORT_3 = """--root='gs://metric'\n\n\nMetric: InboundData
  TotalSamples: 73216
  Accumulator: 64.75GB
Metric: TransferToServerTime
  TotalSamples: 247016
  Accumulator: 1s
Metric: UniqueMetric
  TotalSamples: 9000
  Accumulator: 9000
Counter: CachedSyncTensors
  Value: 1022168
Counter: CreateCompileHandles
  Value: 40
Counter: CreateDataHandles
  Value: 576462152
Counter: UniqueCounter
  Value: 9000
DistractingText
"""

_REPORT_3_SLIGHTLY_DIFFERENT_VALUES = """distracting text\n\n\nMetric: InboundData
  TotalSamples: 70000
  Accumulator: 74.75GB
Metric: TransferToServerTime
  TotalSamples: 247016
  Accumulator: 1s
Metric: UniqueMetric
  TotalSamples: 9000
  Accumulator: 9000
Counter: CachedSyncTensors
  Value: 1022168
Counter: CreateCompileHandles
  Value: 40
Counter: CreateDataHandles
  Value: 576462152
Counter: UniqueCounter
  Value: 9999
DistractingText
"""

_REPORT_3_WITH_NEW_COUNTERS = """Metric: InboundData
  TotalSamples: 73216
  Accumulator: 64.75GB
Metric: TransferToServerTime
  TotalSamples: 247016
  Accumulator: 1s
Metric: UniqueMetric
  TotalSamples: 9000
  Accumulator: 9000
Counter: CachedSyncTensors
  Value: 1022168
Counter: CreateCompileHandles
  Value: 40
Counter: CreateDataHandles
  Value: 576462152
Counter: UniqueCounter
  Value: 9000
Counter: aten::_local_scalar_dense
  Value: 73216
Counter: xla::mean
  Value: 1000
DistractingText
"""


class MetricsCompareUtilsTest(unittest.TestCase):

  def test_get_data_points_from_metrics_reports(self):
    correct_dict = {
        'InboundData__Accumulator_mb': [10000.0, 64750000.0, 64750.0],
        'InboundData__TotalSamples': [1728, 73216, 73216],
        'TransferToServerTime__Accumulator_sec': [89.615, 407467.495546299, 1.0],
        'TransferToServerTime__TotalSamples': [2616, 247016, 247016],
        'CachedSyncTensors__Value': [11336, 1022168, 1022168],
        'CreateCompileHandles__Value': [40, 40, 40],
        'CreateDataHandles__Value': [407992, 576462152, 576462152],
        'UniqueMetric__Accumulator': [None, None, 9000.0],
        'UniqueMetric__TotalSamples': [None, None, 9000],
        'UniqueCounter__Value': [None, None, 9000]
    }
    self.assertEqual(mcu.get_data_points_from_metrics_reports(
        [_REPORT_1, _REPORT_2, _REPORT_3]), correct_dict)


  def test_compare_metrics_reports_no_difference(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3,
        config={'base_tolerance': 2.0})

    # The latest metrics match the previous ones exactly, so the difference
    # report should be empty.
    self.assertEqual(metrics_difference_report, '')


  def test_compare_metrics_reports_value_difference_tolerance_loose(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES,
        config={'base_tolerance': 2.0})

    # Since the tolerance is 2.0, the small differences in values are not
    # big enough to trigger lines in the difference report.
    self.assertEqual(metrics_difference_report, '')


  def test_compare_metrics_reports_value_difference_tolerance_strict(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES,
        config={'base_tolerance': 0.0})

    # Since the tolerance is 0.0, even a tiny difference leads to a line in
    # the metrics_difference_report.
    expected_report = 'InboundData__Accumulator_mb is outside the expected range using tolerance: 0.0. Lower limit: 68083.33333333333  Upper limit: 68083.33333333333  Actual Value: 74750.0\nInboundData__TotalSamples is outside the expected range using tolerance: 0.0. Lower limit: 72144.0  Upper limit: 72144.0  Actual Value: 70000\nUniqueCounter__Value is outside the expected range using tolerance: 0.0. Lower limit: 9333.0  Upper limit: 9333.0  Actual Value: 9999\n'
    self.assertEqual(metrics_difference_report, expected_report)


  def test_compare_metrics_reports_value_difference_tolerance_custom(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES,
        config={
            'base_tolerance': 0.0,
            'InboundData__Accumulator_mb_tolerance': 2.0,
            'UniqueCounter__Value_tolerance': 1.5,
        })

    # 2 of 3 differing values have custom tolerances and therefore should pass.
    # The third uses the default tolerance of 0.0, so it will generate a line.
    expected_report = 'InboundData__TotalSamples is outside the expected range using tolerance: 0.0. Lower limit: 72144.0  Upper limit: 72144.0  Actual Value: 70000\n'
    self.assertEqual(metrics_difference_report, expected_report)
 

  def test_compare_metrics_reports_new_counters(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3_WITH_NEW_COUNTERS,
        config={'base_tolerance': 2.0})

    # Since the tolerance is 2.0, the small differences in values are not
    # big enough to trigger lines in the difference report.
    expected_report = 'Found new aten counter: aten::_local_scalar_dense__Value: 73216\n'
    self.assertEqual(metrics_difference_report, expected_report)


  
  
if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
