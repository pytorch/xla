import os
import sys
import time
import torch_xla_py.xla_model as xm
import torch_xla_py.utils as xu


def _get_device_spec(device):
  ordinal = xm.get_ordinal(defval=-1)
  return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)


# summary_writer should be an instance of torch.utils.tensorborad.SummaryWriter
# or None. If None, no summary files will be written.
def add_scalar_to_summary(summary_writer, metric_name, metric_value,
                          global_step):
  if summary_writer is not None:
    summary_writer.add_scalar(metric_name, metric_value, global_step)


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


def is_first_device(current_device, devices, multiprocess=False):
  """Returns true if this is the first device of the first distributed machine.

  This is useful to know in some cases, e.g. in order to log something that
  does not vary between cores or machines such as learning rate.

  Args:
    current_device: instance of `torch.device`.
    devices: list of device names (strings), e.g. output of
        torch_xla_py.xla_model.get_xla_supported_devices().
  """
  is_first_device = not devices or str(current_device) == devices[0]
  is_first_machine = xm.get_ordinal() == 0
  return is_first_device and is_first_machine

