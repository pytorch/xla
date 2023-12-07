import args_parse

SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
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
    '--persistent_workers': {
        'action': 'store_true',
    },
    '--prefetch_factor': {
        'type': int,
    },
    '--loader_prefetch_size': {
        'type': int,
    },
    '--device_prefetch_size': {
        'type': int,
    },
    '--host_to_device_transfer_threads': {
        'type': int,
    },
    '--sharding': {
        'choices': ['batch', 'spatial', 'conv', 'linear'],
        'nargs': '+',
        'default': [],
    },
    '--use_gradient_checkpointing': {
        'action': 'store_true',
    }
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
import schedulers
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
from torch_xla.distributed.fsdp.wrap import (recursive_wrap,
                                             transformer_auto_wrap_policy)
from torch_xla.distributed.fsdp.utils import checkpoint_module
from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.spmd as xs

DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
    persistent_workers=False,
    prefetch_factor=16,
    loader_prefetch_size=8,
    device_prefetch_size=4,
    num_workers=8,
    host_to_device_transfer_threads=1,
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
                'lr_scheduler_type': 'WarmupAndExponentialDecayScheduler',
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
  print('==> Preparing data..')
  img_dim = get_model_property('img_dim')
  if FLAGS.fake_data:
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
        sample_count=train_dataset_len // FLAGS.batch_size)
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.test_set_batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.test_set_batch_size, dtype=torch.int64)),
        sample_count=50000 // FLAGS.batch_size)
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        drop_last=FLAGS.drop_last,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        persistent_workers=FLAGS.persistent_workers,
        prefetch_factor=FLAGS.prefetch_factor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.test_set_batch_size,
        drop_last=FLAGS.drop_last,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        persistent_workers=FLAGS.persistent_workers,
        prefetch_factor=FLAGS.prefetch_factor)

  torch.manual_seed(42)

  device = xm.xla_device()
  model = get_model_property('model_fn')().to(device)

  if FLAGS.use_gradient_checkpointing:
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            torchvision.models.resnet.BasicBlock,
            torchvision.models.resnet.Bottleneck,
            torchvision.models.vision_transformer.EncoderBlock,
        })
    auto_wrapper_callable = lambda m, *args, **kwargs: checkpoint_module(m)
    model, n_params = recursive_wrap(model, auto_wrap_policy,
                                     auto_wrapper_callable)
    print(f'Wrapped {n_params} parameters for gradient checkpointing.')

  input_mesh = None
  if FLAGS.sharding:
    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    # Model sharding
    if 'conv' in FLAGS.sharding:
      # Shard the model's convlution layers along two dimensions
      mesh_shape = (2, num_devices // 2, 1, 1)
      mesh = xs.Mesh(device_ids, mesh_shape, ('w', 'x', 'y', 'z'))
      partition_spec = (0, 1, 2, 3)  # Apply sharding along all axes
      print(
          f'Applying sharding to convolution layers with mesh {mesh.get_logical_mesh()}'
      )
      for name, layer in model.named_modules():
        if 'conv' in name:
          xs.mark_sharding(layer.weight, mesh, partition_spec)
    elif 'linear' in FLAGS.sharding:
      # Shard the model's fully connected layers across addressable devices
      mesh_shape = (num_devices, 1)
      mesh = xs.Mesh(device_ids, mesh_shape, ('x', 'y'))
      print(
          f'Applying sharding to linear layers with mesh {mesh.get_logical_mesh()}'
      )
      partition_spec = (0, 1)
      for name, layer in model.named_modules():
        if 'fc' in name:
          xs.mark_sharding(layer.weight, mesh, partition_spec)

    # Input sharding
    if 'batch' in FLAGS.sharding or 'spatial' in FLAGS.sharding:
      if 'batch' in FLAGS.sharding and 'spatial' in FLAGS.sharding:
        # Shard along both the batch dimension and spatial dimension
        # If there are more than 4 devices, shard along the height axis as well
        width_axis, height_axis = 2, num_devices // 4
        mesh_shape = (2, 1, width_axis, height_axis)
        input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
        print(
            f'Sharding input along batch and spatial dimensions with mesh {input_mesh.get_logical_mesh()}'
        )
      elif 'batch' in FLAGS.sharding:
        # Shard along batch dimension only
        mesh_shape = (num_devices, 1, 1, 1)
        input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
        print(
            f'Sharding input along batch dimension with mesh {input_mesh.get_logical_mesh()}'
        )
      elif 'spatial' in FLAGS.sharding:
        # Shard two-way along input spatial dimensions
        mesh_shape = (1, 1, num_devices // 2, 2)
        input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
        print(
            f'Sharding input images on spatial dimensions with mesh {input_mesh.get_logical_mesh()}'
        )
      train_loader = pl.MpDeviceLoader(
          train_loader,
          device,
          input_sharding=xs.ShardingSpec(input_mesh, (0, 1, 2, 3)),
          loader_prefetch_size=FLAGS.loader_prefetch_size,
          device_prefetch_size=FLAGS.device_prefetch_size,
          host_to_device_transfer_threads=FLAGS.host_to_device_transfer_threads)

  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(FLAGS.logdir)
  optimizer = optim.SGD(
      model.parameters(),
      lr=FLAGS.lr,
      momentum=FLAGS.momentum,
      weight_decay=1e-4)
  num_training_steps_per_epoch = train_dataset_len // (FLAGS.batch_size)
  lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
      optimizer,
      scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
      scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
      scheduler_divide_every_n_epochs=getattr(
          FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
      num_steps_per_epoch=num_training_steps_per_epoch,
      summary_writer=writer)
  loss_fn = nn.CrossEntropyLoss()

  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):
      x = data.to(xm.xla_device())
      y = target.to(xm.xla_device())
      with xp.StepTrace('train_imagenet'):
        with xp.Trace('build_graph'):
          optimizer.zero_grad()
          if FLAGS.use_gradient_checkpointing:
            for n_l, layer in enumerate(model):
              # Apply gradient checkpointing for reduced memory footprint.
              # This would result in increased computation cost.
              if n_l > 0:
                x = torch_xla.utils.checkpoint.checkpoint(layer, x)
            output = x
          else:
            output = model(x)
          loss = loss_fn(output, y)
          loss.backward()
        optimizer.step()
      xm.mark_step()
      tracker.add(FLAGS.batch_size)
      if lr_scheduler:
        lr_scheduler.step()
      if step % FLAGS.log_steps == 0:
        xm.add_step_closure(
            _train_update, args=(device, step, loss, tracker, epoch, writer))

  def test_loop_fn(loader, epoch):
    total_samples, correct = 0, 0
    model.eval()
    for step, (data, target) in enumerate(loader):
      data = data.to(xm.xla_device())
      target = target.to(xm.xla_device())
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum()
      total_samples += data.size()[0]
      if step % FLAGS.log_steps == 0:
        xm.add_step_closure(
            test_utils.print_test_update, args=(device, None, epoch, step))
    accuracy = 100.0 * correct.item() / total_samples
    return accuracy

  accuracy, max_accuracy = 0.0, 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_loader, epoch)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
    if not FLAGS.test_only_at_end or epoch == FLAGS.num_epochs:
      accuracy = test_loop_fn(test_loader, epoch)
      xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
          epoch, test_utils.now(), accuracy))
      max_accuracy = max(accuracy, max_accuracy)
      test_utils.write_to_summary(
          writer,
          epoch,
          dict_to_write={'Accuracy/test': accuracy},
          write_xla_metrics=True)
    if FLAGS.metrics_debug:
      xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


if __name__ == '__main__':
  if FLAGS.profile:
    server = xp.start_server(FLAGS.profiler_port)

  torch.set_default_dtype(torch.float32)
  accuracy = train_imagenet()
  if accuracy < FLAGS.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  FLAGS.target_accuracy))
    sys.exit(21)
