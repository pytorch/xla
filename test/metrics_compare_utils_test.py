from __future__ import print_function

import unittest

import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.metrics_compare_utils as mcu

_REPORT_1 = """Metric: InboundData
  TotalSamples: 1728
  Accumulator: 10GB
  Rate: 16.8665 / second
  Percentiles: 1%=393.00KB; 5%=393.00KB; 10%=786.00KB; 20%=1.54MB; 50%=1.54MB; 80%=1.54MB; 90%=1.54MB; 95%=1.54MB; 99%=1.54MB
Metric: TransferToServerTime
  TotalSamples: 2616
  Accumulator: 01m29s615ms
  ValueRate: 783ms426.227us / second
  Rate: 24.5054 / second
  Percentiles: 1%=05m003ms; 5%=05m004ms; 10%=05m010ms; 20%=05m015ms; 50%=05m026ms; 80%=05m035ms; 90%=05m082ms; 95%=05m108ms; 99%=05m129ms
Counter: CachedSyncTensors
  Value: 11336
Counter: CreateCompileHandles
  Value: 40
Counter: CreateDataHandles
  Value: 407992"""

_REPORT_2 = """--metric:localhost\n\n\nMetric: InboundData
  TotalSamples: 73216
  Accumulator: 64.75TB
  Percentiles: 1%=393.00KB; 5%=393.00KB; 10%=786.00KB; 20%=1.54MB; 50%=1.54MB; 80%=1.54MB; 90%=1.54MB; 95%=1.54MB; 99%=1.54MB
Metric: TransferToServerTime
  TotalSamples: 247016
  Accumulator: 04d17h11m07s495ms546.299us
  Percentiles: 1%=05m003ms; 5%=05m004ms; 10%=05m010ms; 20%=05m015ms; 50%=05m026ms; 80%=05m035ms; 90%=05m082ms; 95%=05m108ms; 99%=05m129ms
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
  Percentiles: 1%=393.00KB; 5%=393.00KB; 10%=786.00KB; 20%=1.54MB; 50%=1.54MB; 80%=1.54MB; 90%=1.54MB; 95%=1.54MB; 99%=1.54MB
Metric: TransferToServerTime
  TotalSamples: 247016
  Accumulator: 1s
  Percentiles: 1%=05m003ms; 5%=05m004ms; 10%=05m010ms; 20%=05m015ms; 50%=05m026ms; 80%=05m035ms; 90%=05m082ms; 95%=05m108ms; 99%=05m129ms
Metric: UniqueMetric
  TotalSamples: 9000
  Accumulator: 9000
  Percentiles: 1%=8902; 5%=89010; 10%=8920; 20%=8940; 50%=9000; 80%=9060; 90%=9080; 95%=9090; 99%=9098
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
  Percentiles: 1%=393.00KB; 5%=393.00KB; 10%=786.00KB; 20%=1.54MB; 50%=1.54MB; 80%=1.54MB; 90%=1.54MB; 95%=1.54MB; 99%=1.54MB
Metric: TransferToServerTime
  TotalSamples: 247016
  Accumulator: 1s
  Percentiles: 1%=05m003ms; 5%=05m004ms; 10%=05m010ms; 20%=05m015ms; 50%=05m026ms; 80%=05m035ms; 90%=05m082ms; 95%=05m108ms; 99%=05m129ms
Metric: UniqueMetric
  TotalSamples: 9000
  Accumulator: 9000
  Percentiles: 1%=8902; 5%=89010; 10%=8920; 20%=8940; 50%=9000; 80%=9060; 90%=9080; 95%=9090; 99%=9098
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
  Percentiles: 1%=393.00KB; 5%=393.00KB; 10%=786.00KB; 20%=1.54MB; 50%=1.54MB; 80%=1.54MB; 90%=1.54MB; 95%=1.54MB; 99%=1.54MB
Metric: TransferToServerTime
  TotalSamples: 247016
  Accumulator: 1s
  Percentiles: 1%=05m003ms; 5%=05m004ms; 10%=05m010ms; 20%=05m015ms; 50%=05m026ms; 80%=05m035ms; 90%=05m082ms; 95%=05m108ms; 99%=05m129ms
Metric: UniqueMetric
  TotalSamples: 9000
  Accumulator: 9000
  Percentiles: 1%=8902; 5%=89010; 10%=8920; 20%=8940; 50%=9000; 80%=9060; 90%=9080; 95%=9090; 99%=9098
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

  # Return True if 2 dictionaries match using "almost equal" for floats.
  def _dict_almost_equal(self, dict1, dict2):
    if sorted(dict1.keys()) != sorted(dict2.keys()):
      print('dict1 and dict2 had different keys. ({} vs {})'.format(
          sorted(dict1.keys()), sorted(dict2.keys())))
      return False
    for key in dict1:
      values1 = dict1[key]
      values2 = dict2[key]
      if len(values1) != len(values2):
        print('Different number of values for key {}. ({} vs {}).'.format(
            key, len(values1), len(values2)))
        return False
      for v1, v2 in zip(values1, values2):
        if v1 is None and v2 is None:
          continue
        else:
          try:
            # Check that all numeric values are close enough.
            if abs(v1 - v2) > max(1e-12 * max(abs(v1), abs(v2)), 0.0):
              print('Floats not close enough for key {}. ({} vs {})'.format(
                  key, values1, values2))
              return False
          except TypeError:
            print('None values differed for key {}. ({} vs {})'.format(
                key, values1, values2))
            return False
    return True


  def test_get_data_points_from_metrics_reports(self):
    correct_dict = {
        'InboundData__TotalSamples': [1728.0, 73216.0, 73216.0],
        'InboundData__Accumulator_mb': [10000.0, 64750000.0, 64750.0],
        'InboundData__Percentile_1_mb': [0.393, 0.393, 0.393],
        'InboundData__Percentile_5_mb': [0.393, 0.393, 0.393],
        'InboundData__Percentile_10_mb': [0.786, 0.786, 0.786],
        'InboundData__Percentile_20_mb': [1.54, 1.54, 1.54],
        'InboundData__Percentile_50_mb': [1.54, 1.54, 1.54],
        'InboundData__Percentile_80_mb': [1.54, 1.54, 1.54],
        'InboundData__Percentile_90_mb': [1.54, 1.54, 1.54],
        'InboundData__Percentile_95_mb': [1.54, 1.54, 1.54],
        'InboundData__Percentile_99_mb': [1.54, 1.54, 1.54],
        'TransferToServerTime__TotalSamples': [2616.0, 247016.0, 247016.0],
        'TransferToServerTime__Accumulator_sec': [89.615, 407467.495546299, 1.0],
        'TransferToServerTime__Percentile_1_sec': [300.003, 300.003, 300.003],
        'TransferToServerTime__Percentile_5_sec': [300.004, 300.004, 300.004],
        'TransferToServerTime__Percentile_10_sec': [300.01, 300.01, 300.01],
        'TransferToServerTime__Percentile_20_sec': [300.015, 300.015, 300.015],
        'TransferToServerTime__Percentile_50_sec': [300.026, 300.026, 300.026],
        'TransferToServerTime__Percentile_80_sec': [300.035, 300.035, 300.035],
        'TransferToServerTime__Percentile_90_sec': [300.082, 300.082, 300.082],
        'TransferToServerTime__Percentile_95_sec': [300.108, 300.108, 300.108],
        'TransferToServerTime__Percentile_99_sec': [300.129, 300.129, 300.129],
        'UniqueMetric__TotalSamples': [None, None, 9000.0],
        'UniqueMetric__Accumulator': [None, None, 9000.0],
        'UniqueMetric__Percentile_1': [None, None, 8902.0],
        'UniqueMetric__Percentile_5': [None, None, 89010.0],
        'UniqueMetric__Percentile_10': [None, None, 8920.0],
        'UniqueMetric__Percentile_20': [None, None, 8940.0],
        'UniqueMetric__Percentile_50': [None, None, 9000.0],
        'UniqueMetric__Percentile_80': [None, None, 9060.0],
        'UniqueMetric__Percentile_90': [None, None, 9080.0],
        'UniqueMetric__Percentile_95': [None, None, 9090.0],
        'UniqueMetric__Percentile_99': [None, None, 9098.0],
        'CachedSyncTensors__Value': [11336, 1022168, 1022168],
        'CreateCompileHandles__Value': [40, 40, 40],
        'CreateDataHandles__Value': [407992, 576462152, 576462152],
        'UniqueCounter__Value': [None, None, 9000]
    }

    self.assertTrue(self._dict_almost_equal(
        mcu.get_data_points_from_metrics_reports(
            [_REPORT_1, _REPORT_2, _REPORT_3]),
        correct_dict))


  def test_compare_metrics_reports_no_difference(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3,
        config={'base_expression': 'v == v_mean'})

    # The latest metrics match the previous ones exactly, so the difference
    # report should be empty.
    self.assertEqual(metrics_difference_report, '')


  def test_compare_metrics_reports_value_difference_tolerance_loose(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES,
        config={'base_expression': 'v <= v_mean + (v_stddev * 2.0)'})

    # Since the tolerance is 2.0, the small differences in values are not
    # big enough to trigger lines in the difference report.
    self.assertEqual(metrics_difference_report, '')


  def test_compare_metrics_reports_value_difference_tolerance_strict(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES,
        config={'base_expression': 'v == v_mean'})

    # Since the tolerance is 0.0, even a tiny difference leads to a line in
    # the metrics_difference_report.
    expected_report = 'InboundData__Accumulator_mb failed its expression check. Expression: v == v_mean.  Mean: 68083.33333333333.  Stddev: 4714.045207910317.  Actual Value: 74750.0\nInboundData__TotalSamples failed its expression check. Expression: v == v_mean.  Mean: 72144.0.  Stddev: 1516.0369388639579.  Actual Value: 70000.0\nUniqueCounter__Value failed its expression check. Expression: v == v_mean.  Mean: 9333.0.  Stddev: 470.93311627024065.  Actual Value: 9999\n'
    self.assertEqual(metrics_difference_report, expected_report)


  def test_compare_metrics_reports_value_difference_tolerance_custom(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES,
        config={
            'base_expression': 'True',
            'InboundData__Accumulator_mb_expression': 'v < v_mean',
            'UniqueCounter__Value_expression': 'v < v_mean',
        })

    # 2 of 3 differing values have custom tolerances and therefore should pass.
    # The third uses the default tolerance of 0.0, so it will generate a line.
    expected_report = 'InboundData__Accumulator_mb failed its expression check. Expression: v < v_mean.  Mean: 68083.33333333333.  Stddev: 4714.045207910317.  Actual Value: 74750.0\nUniqueCounter__Value failed its expression check. Expression: v < v_mean.  Mean: 9333.0.  Stddev: 470.93311627024065.  Actual Value: 9999\n'
    self.assertEqual(metrics_difference_report, expected_report)
 

  def test_compare_metrics_reports_new_counters(self):
    data_points = mcu.get_data_points_from_metrics_reports(
        [_REPORT_3, _REPORT_3, _REPORT_3_SLIGHTLY_DIFFERENT_VALUES])
    metrics_difference_report = mcu.compare_metrics(
        data_points, _REPORT_3_WITH_NEW_COUNTERS,
        config={'base_expression': 'v <= v_mean + (v_stddev * 2.0)'})

    # Since the tolerance is 2.0, the small differences in values are not
    # big enough to trigger lines in the difference report.
    expected_report = 'Found new aten counter: aten::_local_scalar_dense__Value: 73216\n'
    self.assertEqual(metrics_difference_report, expected_report)

  def test_parse_real_metrics(self):
    print("Testing against TPU. If this hangs, check that $XRT_TPU_CONFIG is set")
    x = torch.rand(3, 5, device=xm.xla_device())
    x = torch.flatten(x, 1)
    x = torch.roll(x, 1, 0)
    x = torch.flip(x, [0, 1])
    self.assertEqual(x.device.type, 'xla')
    metrics = met.metrics_report()
    self.assertTrue(metrics)
    data_points = mcu.get_data_points_from_metrics_reports([metrics])
    self.assertIn('CompileTime__Percentile_99_sec', data_points.keys())
    self.assertIn('CompileTime__TotalSamples', data_points.keys())
  

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
