import test_utils

ALEXNET = 'alexnet'
DENSENET121 = 'densenet121'
DENSENET161 = 'densenet161'
DENSENET169 = 'densenet169'
DENSENET201 = 'densenet201'
INCEPTION_V3 = 'inception_v3'
RESNET101 = 'resnet101'
RESNET152 = 'resnet152'
RESNET18 = 'resnet18'
RESNET34 = 'resnet34'
RESNET50 = 'resnet50'
SQUEEZENET1_0 = 'squeezenet1_0'
SQUEEZENET1_1 = 'squeezenet1_1'
VGG11 = 'vgg11'
VGG11_BN = 'vgg11_bn'
VGG13 = 'vgg13'
VGG13_BN = 'vgg13_bn'
VGG16 = 'vgg16'
VGG16_BN = 'vgg16_bn'
VGG19 = 'vgg19'
VGG19_BN = 'vgg19_bn'
SUPPORTED_MODELS = [
    ALEXNET,
    DENSENET121,
    DENSENET161,
    DENSENET169,
    DENSENET201,
    INCEPTION_V3,
    RESNET101,
    RESNET152,
    RESNET18,
    RESNET34,
    RESNET50,
    #SQUEEZENET1_0,
    #SQUEEZENET1_1,
    VGG11,
    VGG11_BN,
    VGG13,
    VGG13_BN,
    VGG16,
    VGG16_BN,
    VGG19,
    VGG19_BN
]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': RESNET50,
    }
}
FLAGS = test_utils.parse_common_options(
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
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import unittest

DEFAULT_KWARGS = dict(
    batch_size=128,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    RESNET50: DEFAULT_KWARGS,
    INCEPTION_V3: DEFAULT_KWARGS,
}

default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)


def load_torchvision_model(model_name):
  # inception requires special treatment since it outputs a tuple by default
  if model_name.startswith('inception'):
    model = lambda: torchvision.models.inception_v3(aux_logits=False)
  else:
    model = getattr(torchvision.models, FLAGS.model)
  return model


def train_imagenet():
  print('==> Preparing data..')
  img_dim = 299 if FLAGS.model.startswith('inception') else 224
  if FLAGS.fake_data:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
        sample_count=1200000 // FLAGS.batch_size)
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers)
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'val'),
        transforms.Compose([
            transforms.RandomResizedCrop(img_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  devices = xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)
  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  torchvision_model = load_torchvision_model(FLAGS.model)
  model_parallel = dp.DataParallel(torchvision_model, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=5e-4)
    tracker = xm.RateTracker()
    for x, (data, target) in loader:
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(FLAGS.batch_size)
      if x % FLAGS.log_steps == 0:
        print('[{}]({}) Loss={:.5f} Rate={:.2f}'.format(device, x, loss.item(),
                                                        tracker.rate()))

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    for x, (data, target) in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      total_samples += data.size()[0]

    print('[{}] Accuracy={:.2f}%'.format(device,
                                         100.0 * correct / total_samples))
    return correct / total_samples

  accuracy = 0.0
  for epoch in range(1, FLAGS.num_epochs + 1):
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = sum(accuracies) / len(devices)
    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  return accuracy * 100.0


class TrainImageNet(TestCase):

  def tearDown(self):
    super(TrainImageNet, self).tearDown()

  def test_accurracy(self):
    self.assertGreaterEqual(train_imagenet(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
