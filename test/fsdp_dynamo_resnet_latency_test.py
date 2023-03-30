import args_parse
from functools import partial

SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16'
]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'resnet50',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
    },
    '--lr_scheduler_divisor': {
        'type': int,
    },
    '--test_only_at_end': {
        'action': 'store_true',
    },
    '--num_warmup_epochs': {
        'type': float,
        'default': 0.9,
    },
    '--eval_interval': {
        'type': int,
        'default': 1,
    },
    '--flatten_parameters': {
        'action': 'store_true',
    },
    '--auto_wrap_policy': {
        'choices': ['none', 'size_based', 'type_based'],
        'default': 'none',
    },
    '--auto_wrap_min_num_params': {
        'type': int,
        'default': 1e6,
    },
    '--use_nested_fsdp': {
        'action': 'store_true',
    },
    '--use_gradient_checkpointing': {
        'action': 'store_true',
    },
    '--compute_dtype': {
        'choices': ['float32', 'float16', 'bfloat16'],
        'default': 'float32',
    },
    '--fp32_reduce_scatter': {
        'action': 'store_true',
    },
    '--shard_param_on_dim_0': {
        'action': 'store_true',
    },
    '--no_pin_layout_in_collective_ops': {
        'action': 'store_false',
        'dest': 'pin_layout_in_collective_ops',
    },
    '--use_small_fake_sample': {
        'action': 'store_true',
    },
    '--profile': {
        'action': 'store_true',
    },
    '--use_fsdp': {
        'action': 'store_true',
    },
    '--use_dynamo': {
        'action': 'store_true',
    },
    '--print_metrics': {
        'action': 'store_true',
    },
    '--do_train': {
        'action': 'store_true',
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
)

import os
import sys
import schedulers
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
from torch_xla.distributed.fsdp.wrap import (size_based_auto_wrap_policy,
                                             transformer_auto_wrap_policy)

DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
    'resnet50':
        dict(
            DEFAULT_KWARGS, **{
                'lr': 0.5,
                'lr_scheduler_divide_every_n_epochs': 20,
                'lr_scheduler_divisor': 5,
            })
}

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)


def get_model_property(key):
  default_model_property = {
      'img_dim': 224,
      'model_fn': getattr(torchvision.models, FLAGS.model)
  }
  model_properties = {
      'inception_v3': {
          'img_dim': 299,
          'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
      },
  }
  model_fn = model_properties.get(FLAGS.model, default_model_property)[key]
  return model_fn


def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train_imagenet():
  start_cold = time.time()
  print('==> Preparing data..')
  img_dim = get_model_property('img_dim')
  if FLAGS.fake_data:
    use_small_fake_sample = FLAGS.use_small_fake_sample
    train_dataset_len = 50000 if use_small_fake_sample else 1200000  # Roughly the size of Imagenet dataset.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
        sample_count=train_dataset_len // FLAGS.batch_size //
        xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.test_set_batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.test_set_batch_size, dtype=torch.int64)),
        sample_count=50000 // FLAGS.batch_size // xm.xrt_world_size())
  else:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(img_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset_len = len(train_dataset.imgs)
    resize_dim = max(img_dim, 256)
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'val'),
        # Matches Torchvision's eval transforms except Torchvision uses size
        # 256 resize for all models both here and in the train loader. Their
        # version crashes during training on 299x299 images, e.g. inception.
        transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(img_dim),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler, test_sampler = None, None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
      test_sampler = torch.utils.data.distributed.DistributedSampler(
          test_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False if train_sampler else True,
        persistent_workers=True,
        num_workers=FLAGS.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.test_set_batch_size,
        sampler=test_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False,
        persistent_workers=True,
        num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  device = xm.xla_device()
  model = get_model_property('model_fn')()
  # Automatic wrapping sub-modules with inner FSDP
  auto_wrap_policy = None
  auto_wrapper_callable = None
  if FLAGS.auto_wrap_policy != "none":
    if FLAGS.auto_wrap_policy == "size_based":
      # auto-wrap all sub-modules with a certain number of parameters (default 1e6)
      auto_wrap_policy = partial(
          size_based_auto_wrap_policy,
          min_num_params=FLAGS.auto_wrap_min_num_params)
    elif FLAGS.auto_wrap_policy == "type_based":
      # auto-wrap all sub-modules in torchvision ResNet's BasicBlock or Bottleneck
      # or torchvision transformer's EncoderBlock as an example
      # (transformer_auto_wrap_policy wraps all sub-modules in transformer_layer_cls)
      auto_wrap_policy = partial(
          transformer_auto_wrap_policy,
          transformer_layer_cls={
              torchvision.models.resnet.BasicBlock,
              torchvision.models.resnet.Bottleneck,
              torchvision.models.vision_transformer.EncoderBlock,
          })
    else:
      raise Exception(f"Invalid auto-wrap policy: {FLAGS.auto_wrap_policy}")
    if FLAGS.use_gradient_checkpointing:
      # Apply gradient checkpointing to auto-wrapped sub-modules if specified
      auto_wrapper_callable = lambda m, *args, **kwargs: FSDP(
          checkpoint_module(m), *args, **kwargs)

  fsdp_wrap = lambda m: FSDP(
      m,
      compute_dtype=getattr(torch, FLAGS.compute_dtype),
      fp32_reduce_scatter=FLAGS.fp32_reduce_scatter,
      flatten_parameters=FLAGS.flatten_parameters,
      shard_param_on_dim_0=FLAGS.shard_param_on_dim_0,
      pin_layout_in_collective_ops=FLAGS.pin_layout_in_collective_ops,
      auto_wrap_policy=auto_wrap_policy,
      auto_wrapper_callable=auto_wrapper_callable,
      optimization_barrier_in_forward=False,
      optimization_barrier_in_backward=False)
  
  # Always wrap the base model with an outer FSDP
  if(FLAGS.use_fsdp):
    model = fsdp_wrap(model)
  else:
    model.to(device)

  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(FLAGS.logdir)
  optimizer = optim.SGD(
      model.parameters(),
      lr=FLAGS.lr,
      momentum=FLAGS.momentum,
      weight_decay=1e-4)
  num_training_steps_per_epoch = train_dataset_len // (
      FLAGS.batch_size * xm.xrt_world_size())
  lr_scheduler = schedulers.WarmupAndExponentialDecayScheduler(
      optimizer,
      num_steps_per_epoch=num_training_steps_per_epoch,
      divide_every_n_epochs=FLAGS.lr_scheduler_divide_every_n_epochs,
      divisor=FLAGS.lr_scheduler_divisor,
      num_warmup_epochs=FLAGS.num_warmup_epochs,
      summary_writer=writer)
  loss_fn = nn.CrossEntropyLoss()

  if FLAGS.profile:
    server = xp.start_server(FLAGS.profiler_port)

  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):
      with xp.StepTrace('train_imagenet', step_num=step):
        with xp.Trace('build_graph'):
          optimizer.zero_grad()
          output = model(data)
          loss = loss_fn(output, target)
          loss.backward()
          optimizer.step()  # do not reduce gradients on sharded params
          tracker.add(FLAGS.batch_size)
          if lr_scheduler:
            lr_scheduler.step()
        if step % FLAGS.log_steps == 0:
          xm.add_step_closure(
              _train_update, args=(device, step, loss, tracker, epoch, writer))

  def inference_loop_fn(loader, epoch):
    model.eval()
    if(FLAGS.use_dynamo):
      model = torch.compile(model, backend='torchxla_trace_once')
    for step, (data, target) in enumerate(loader):
      if(step == 1):
        start_warm = time.time()
      output = model(data)
    return start_warm

  if(FLAGS.do_train):
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    for epoch in range(1, FLAGS.num_epochs + 1):
      xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
      train_loop_fn(train_device_loader, epoch)
      xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
  
  print('Starting inference...')
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  with torch.no_grad():
    start_warm = inference_loop_fn(test_device_loader, epoch)
  end = time.time()
  print('Done.')
  sample_count_per_device = float(FLAGS.sample_count)/xm.xrt_world_size()
  elapsed_time_cold = end-start_cold;
  elapsed_time_warm = end-start_warm;
  elapsed_time_cold_per_sample = elapsed_time_cold/sample_count_per_device*1000
  elapsed_time_warm_per_sample = elapsed_time_warm/max(sample_count_per_device-1, 1)*1000
  print(f'Total cold time (s): {elapsed_time_cold} for {sample_count_per_device} samples')
  print(f'Total cold per sample (ms): {elapsed_time_cold_per_sample}')
  print(f'Total warm time (s): {elapsed_time_warm} for {sample_count_per_device-1} samples')
  print(f'Total warm per sample (ms): {elapsed_time_warm_per_sample}')

  if FLAGS.print_metrics:
      xm.master_print(met.metrics_report(), flush=True)

  test_utils.close_summary_writer(writer)
  return 100


def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_imagenet()


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
