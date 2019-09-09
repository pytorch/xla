# This module cannot import any other PyTorch/XLA module. Only Python core modules.
import argparse
import os
import sys
import time


# summary_writer should be an instance of torch.utils.tensorborad.SummaryWriter
# or None. If None, no summary files will be written.
def add_scalar_to_summary(summary_writer, metric_name, metric_value,
                          global_step):
  if summary_writer is not None:
    summary_writer.add_scalar(metric_name, metric_value, global_step)


def parse_common_options(datadir=None,
                         logdir=None,
                         num_cores=None,
                         batch_size=128,
                         num_epochs=10,
                         num_workers=4,
                         log_steps=20,
                         lr=None,
                         momentum=None,
                         target_accuracy=None,
                         opts=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--datadir', type=str, default=datadir)
  parser.add_argument('--logdir', type=str, default=logdir)
  parser.add_argument('--lr_scheduler_type', type=str, default=None)
  parser.add_argument('--num_cores', type=int, default=num_cores)
  parser.add_argument('--batch_size', type=int, default=batch_size)
  parser.add_argument('--num_epochs', type=int, default=num_epochs)
  parser.add_argument('--num_workers', type=int, default=num_workers)
  parser.add_argument('--log_steps', type=int, default=log_steps)
  parser.add_argument('--lr_scheduler_divide_every_n_epochs', type=int)
  parser.add_argument('--lr_scheduler_divisor', type=int)
  parser.add_argument('--lr', type=float, default=lr)
  parser.add_argument('--momentum', type=float, default=momentum)
  parser.add_argument('--target_accuracy', type=float, default=target_accuracy)
  parser.add_argument('--fake_data', action='store_true')
  parser.add_argument('--tidy', action='store_true')
  parser.add_argument('--metrics_debug', action='store_true')
  if opts:
    for name, aopts in opts:
      parser.add_argument(name, **aopts)
  args, leftovers = parser.parse_known_args()
  sys.argv = [sys.argv[0]] + leftovers
  # Setup import folders.
  xla_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
  sys.path.append(os.path.join(os.path.dirname(xla_folder), 'test'))
  sys.path.insert(0, xla_folder)
  return args


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
      device, step_num, loss, rate, global_rate, time.asctime()))


def print_test_update(device, accuracy):
  """Prints single-core test metrics.

  Args:
    device: Instance of `torch.device`.
    accuracy: Float.
  """
  print('[{}] Accuracy={:.2f}%'.format(device, accuracy))


def should_report_lr(current_device, devices, machine_ordinal_num):
  """Returns true if this device should log the learning rate.

  In order to avoid many duplicate copies of the same learning rate in the
  Tensorboard metrics, we only log the learning rate of a single representative
  device. Use the device and machine number to uniquely identify this
  representative.

  Args:
    current_device: instance of `torch.device`.
    devices: list of device names (strings), e.g. output of
        torch_xla_py.xla_model.get_xla_supported_devices().
    machine_ordinal_num: int, output of torch_xla_py.xla_model.get_ordinal().
  """
  is_first_device = not devices or str(current_device) == devices[0]
  is_first_machine = machine_ordinal_num == 0
  return is_first_device and is_first_machine

