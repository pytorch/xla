import test_utils

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
MODEL_SPECIFIC_DEFAULTS = {}

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
        # Matches Torchvision's eval transforms except Torchvision uses size
        # 256 resize for all models and just crashes for 299x299 images, e.g.
        # inception_v3.
        transforms.Compose([
            transforms.Resize(max(img_dim, 256)),
            transforms.CenterCrop(img_dim),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
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
        print('[{}]({}) Loss={:.5f} Rate={:.2f}'.format(device, x, loss.item(),
                                                        tracker.rate()))

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    model.eval()
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
    accuracy = sum(accuracies) / len(accuracies)
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
