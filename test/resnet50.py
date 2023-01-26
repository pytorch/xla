import args_parse

MODEL_OPTS = {
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
    '--sharded': {
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
import torch_xla.test.test_utils as test_utils
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharding import Mesh
import time

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
                'lr_scheduler_type': 'WarmupAndExponentialDecayScheduler',
            })
}

IMG_DIM = 224

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get('resnet50', DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)


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
  img_dim = IMG_DIM
  if FLAGS.fake_data:
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    train_dataset_len = 1200
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
        sampler=None,
        drop_last=FLAGS.drop_last,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.test_set_batch_size,
        sampler=None,
        drop_last=FLAGS.drop_last,
        shuffle=False)

  torch.manual_seed(42)

  device = xm.xla_device()

  num_devices = len(xm.get_xla_supported_devices())
  print('num_devices:', num_devices)
  mesh_shape = (num_devices, 1, 1, 1)
  #mesh_shape = (1, 1, 4, 2)
  device_ids = np.array(range(num_devices))
  mesh = Mesh(device_ids, mesh_shape, ('x', 'y', 'z', 'w'))

  model = torchvision.models.resnet50().to(device)
  #if FLAGS.sharded:
  #for name, layer in model.named_modules():
  #if 'conv' in name:
  #print('Attempting to apply sharding to', name)
  #partition = int(xm.xrt_world_size() / 2)
  #xs.mark_sharding(layer.weight, mesh, (0, 1, 2, 3))
  input("ready?")
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
  lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
      optimizer,
      scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
      scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
      scheduler_divide_every_n_epochs=getattr(
          FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
      num_steps_per_epoch=num_training_steps_per_epoch,
      summary_writer=writer)
  loss_fn = nn.CrossEntropyLoss()

  server = xp.start_server(FLAGS.profiler_port)

  def train_loop_fn(loader, epoch):
    start_time = time.time()
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):

      data = data.to(xm.xla_device())
      #TODO: CreateTensorNode crashes
      xs.mark_sharding(data, mesh, (0, 1, 2, 3))
      target = target.to(xm.xla_device())
      with xp.StepTrace('train_imagenet'):
        with xp.Trace('build_graph'):
          optimizer.zero_grad()
          output = model(data)
          print('output computed')
          loss = loss_fn(output, target)
          print('loss computed')
          loss.backward()
        optimizer.step()
        print('optmizer stepped')
        tracker.add(FLAGS.batch_size)
        if lr_scheduler:
          lr_scheduler.step()
        if step % FLAGS.log_steps == 0:
          xm.add_step_closure(
              _train_update, args=(device, step, loss, tracker, epoch, writer))
      xm.mark_step()
      #if step == 0:
      #    start_time = time.time()
      print('mark stepped')
    end_time = time.time()
    print(f"epoch with {step+1} steps finished in {end_time-start_time}")

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

  train_device_loader = train_loader
  test_device_loader = test_loader
  accuracy, max_accuracy = 0.0, 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader, epoch)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
    if not FLAGS.test_only_at_end or epoch == FLAGS.num_epochs:
      accuracy = test_loop_fn(test_device_loader, epoch)
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
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_imagenet()
  if accuracy < FLAGS.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  FLAGS.target_accuracy))
    sys.exit(21)
