import argparse
import os
import sys
import torch_xla_py.xla_model as xm


def parse_common_options(datadir=None,
                         logdir=None,
                         num_cores=1,
                         batch_size=128,
                         num_epochs=10,
                         num_workers=4,
                         target_accuracy=None,
                         opts=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--datadir', type=str, default=datadir)
  parser.add_argument('--logdir', type=str, default=logdir)
  parser.add_argument('--num_cores', type=int, default=num_cores)
  parser.add_argument('--batch_size', type=int, default=batch_size)
  parser.add_argument('--num_epochs', type=int, default=num_epochs)
  parser.add_argument('--num_workers', type=int, default=num_workers)
  parser.add_argument('--target_accuracy', type=float, default=target_accuracy)
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


def _get_summary_writer(logdir=None):
  if logdir:
    from tensorboardX import SummaryWriter
    return SummaryWriter(logdir)


def get_log_fn(logdir=None, custom_log_fn=print):
  writer = _get_summary_writer(logdir)

  def log_fn(step_result):
    if (isinstance(step_result, xm.TrainStepMetrics) or
        isinstance(step_result, xm.TestStepMetrics)):
      step_result.write_summary(writer)
      custom_log_fn(step_result.log_str())
    else:
      custom_log_fn(step_result)

  return log_fn
