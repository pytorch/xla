#!/usr/bin/env python

from __future__ import print_function

import argparse
import threading
import torch
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm


def run_benchmark(args, pos_args):
  devices = xm.get_xla_supported_devices(max_devices=args.max_devices)
  shape = [int(x) for x in args.shape.split(',')]

  send_list = []
  for i in range(0, len(devices)):
    mb = []
    for j in range(0, args.prefetch):
      mb.append(torch.randn(*shape))
    send_list.append(mb)

  def threadfn(i):
    device = devices[i]
    xdevices = [device] * len(send_list[i])
    for n in range(0, args.test_count):
      with xu.TimedScope(msg='Send[{}][{}]: '.format(i, n), printfn=print):
        _ = torch_xla._XLAC._xla_tensors_from_aten(send_list[i], xdevices)

  threads = []
  for i in range(0, len(devices)):
    t = threading.Thread(target=threadfn, args=(i,))
    t.start()
    threads.append(t)
  for t in threads:
    t.join()
  print(met.metrics_report())


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--test_count', type=int, default=20)
  arg_parser.add_argument('--prefetch', type=int, default=4)
  arg_parser.add_argument('--max_devices', type=int, default=None)
  # Same size as resnet50 bs=128 but avoid re-layout to drop tensor transform cost.
  arg_parser.add_argument('--shape', type=str, default='384,224,224')
  args, pos_args = arg_parser.parse_known_args()
  run_benchmark(args, pos_args)
