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
        'default': 64,
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
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    opts=MODEL_OPTS.items(),
)

from common_utils import TestCase, run_tests
import os
import schedulers
from statistics import mean
import test_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import unittest

DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    'resnet50': dict({
        'lr_scheduler_divide_every_n_epochs': 20,
        'lr_scheduler_divisor': 5,
        'lr_scheduler_type': 'WarmupAndExponentialDecayScheduler',
      }, **DEFAULT_KWARGS)
}

default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)

MODEL_PROPERTIES = {
    'inception_v3': {
        'img_dim': 299,
        'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
    },
    'DEFAULT': {
        'img_dim': 224,
        'model_fn': getattr(torchvision.models, FLAGS.model)
    }
}


def get_model_property(key):
  return MODEL_PROPERTIES.get(FLAGS.model, MODEL_PROPERTIES['DEFAULT'])[key]


def train_imagenet():
  print('==> Preparing data..')
  img_dim = get_model_property('img_dim')
  if FLAGS.fake_data:
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
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

    train_sampler = None
    test_sampler = None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset, num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(), shuffle=True)
      test_sampler = torch.utils.data.distributed.DistributedSampler(
          test_dataset, num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(), shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=FLAGS.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.test_set_batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  devices = (
      xm.get_xla_supported_devices(
          max_devices=FLAGS.num_cores) if FLAGS.num_cores != 0 else [])
  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  torchvision_model = get_model_property('model_fn')
  model_parallel = dp.DataParallel(torchvision_model, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = context.getattr_or(
        'optimizer', lambda: optim.SGD(
            model.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
            weight_decay=5e-4))
    lr_scheduler = context.getattr_or(
        'lr_scheduler', lambda: schedulers.wrap_optimizer_with_scheduler(
            optimizer,
            scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
            scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
            scheduler_divide_every_n_epochs=getattr(
                FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
            num_steps_per_epoch=num_training_steps_per_epoch,
            summary_writer=writer if test_utils.is_first_device(
                device, devices) else None))
    tracker = xm.RateTracker()
    model.train()
    for x, (data, target) in loader:
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(FLAGS.batch_size)
      if x % FLAGS.log_steps == 0:
        test_utils.print_training_update(device, x, loss.item(),
                                         tracker.rate(),
                                         tracker.global_rate())
      if lr_scheduler:
        lr_scheduler.step()

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    model.eval()
    for x, (data, target) in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      total_samples += data.size()[0]

    accuracy = 100.0 * correct / total_samples
    test_utils.print_test_update(device, accuracy)
    return accuracy

  accuracy = 0.0
  writer = SummaryWriter(log_dir=FLAGS.logdir) if FLAGS.logdir else None
  num_devices = len(
      xm.xla_replication_devices(devices)) if len(devices) > 1 else 1
  num_training_steps_per_epoch = train_dataset_len // (
      FLAGS.batch_size * num_devices)
  for epoch in range(1, FLAGS.num_epochs + 1):
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = mean(accuracies)
    print("Epoch: {}, Mean Accuracy: {:.2f}%".format(epoch, accuracy))
    global_step = (epoch - 1) * num_training_steps_per_epoch
    test_utils.add_scalar_to_summary(writer, 'Accuracy/test', accuracy,
                                     global_step)
    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  return accuracy


class TrainImageNet(TestCase):

  def tearDown(self):
    super(TrainImageNet, self).tearDown()

  def test_accurracy(self):
    self.assertGreaterEqual(train_imagenet(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
