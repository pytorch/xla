import os
import sys
import time
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu


def _get_device_spec(device):
  ordinal = xm.get_ordinal(defval=-1)
  return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)


def add_scalar_to_summary(summary_writer, metric_name, metric_value,
                          global_step):
  """Add a scalar data point to a SummaryWriter.

  Args:
    summary_writer: instance of Tensorboard SummaryWriter or
        None. If None, no summary files will be written.
    metric_name: Str. Name of the metric to write.
    metric_value: Float. Value of the metric.
    global_step: Int. The global step value for this data point.
  """
  if summary_writer is not None:
    summary_writer.add_scalar(metric_name, metric_value, global_step)


def close_summary_writer(summary_writer):
  """Flush and close a SummaryWriter.

  Args:
    summary_writer: instance of Tensorboard SummaryWriter or
        None. If None, no action is taken.
  """
  if summary_writer is not None:
    summary_writer.flush()
    summary_writer.close()


def get_summary_writer(logdir):
  """Initialize a Tensorboard SummaryWriter.

  Args:
    logdir: Str. File location where logs will be written or None. If None,
        no writer is created.

  Returns:
    Instance of Tensorboard SummaryWriter.
  """
  if logdir:
    from tensorboardX import SummaryWriter
    return SummaryWriter(log_dir=logdir)


def print_training_update(device, step_num, loss, rate, global_rate):
  """Prints the training metrics at a given step.

  Args:
    device: Instance of `torch.device`.
    step_num: Integer. Current step number.
    loss: Float. Current loss.
    rate: Float. The examples/sec rate for the current batch.
    global_rate: Float. The average examples/sec rate since training began.
  """
  print('[{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
      _get_device_spec(device), step_num, loss, rate, global_rate,
      time.asctime()))


def print_test_update(device, accuracy):
  """Prints single-core test metrics.

  Args:
    device: Instance of `torch.device`.
    accuracy: Float.
  """
  print('[{}] Accuracy={:.2f}%'.format(_get_device_spec(device), accuracy))
