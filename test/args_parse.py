# This module cannot import any other PyTorch/XLA module. Only Python core modules.
import argparse
import os
import sys


def parse_common_options(datadir=None,
                         logdir=None,
                         num_cores=None,
                         batch_size=128,
                         num_epochs=10,
                         num_workers=4,
                         prefetch_factor=8,
                         pin_memory=False,
                         persistent_workers=True,
                         loader_prefetch_size=8,
                         device_prefetch_size=4,
                         cpu_to_device_transfer_threads=1,
                         log_steps=20,
                         lr=None,
                         momentum=None,
                         target_accuracy=None,
                         profiler_port=9012,
                         opts=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument('--datadir', type=str, default=datadir)
  parser.add_argument('--logdir', type=str, default=logdir)
  parser.add_argument('--num_cores', type=int, default=num_cores)
  parser.add_argument('--batch_size', type=int, default=batch_size)
  parser.add_argument('--num_epochs', type=int, default=num_epochs)
  parser.add_argument('--num_workers', type=int, default=num_workers)
  parser.add_argument("--prefetch_factor", type=int, default=prefetch_factor)
  parser.add_argument('--pin_memory', type=bool, default=pin_memory)
  parser.add_argument('--persistent_workers', type=int, default=persistent_workers)
  parser.add_argument("--loader_prefetch_size", type=int, default=loader_prefetch_size)
  parser.add_argument("--device_prefetch_size", type=int, default=device_prefetch_size)
  parser.add_argument("--cpu_to_device_transfer_threads", type=int, default=cpu_to_device_transfer_threads)
  parser.add_argument('--log_steps', type=int, default=log_steps)
  parser.add_argument('--profiler_port', type=int, default=profiler_port)
  parser.add_argument('--lr', type=float, default=lr)
  parser.add_argument('--momentum', type=float, default=momentum)
  parser.add_argument('--target_accuracy', type=float, default=target_accuracy)
  parser.add_argument('--drop_last', action='store_true')
  parser.add_argument('--fake_data', action='store_true')
  parser.add_argument('--tidy', action='store_true')
  parser.add_argument('--metrics_debug', action='store_true')
  parser.add_argument('--async_closures', action='store_true')
  parser.add_argument('--debug', action='store_true')
  if opts:
    for name, aopts in opts:
      parser.add_argument(name, **aopts)
  args, leftovers = parser.parse_known_args()
  sys.argv = [sys.argv[0]] + leftovers
  # Setup import folders.
  xla_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
  sys.path.append(os.path.join(os.path.dirname(xla_folder), 'test'))
  return args

