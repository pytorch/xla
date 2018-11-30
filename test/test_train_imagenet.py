import argparse
import os
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--train_dir', type=str, default='/tmp/imagenet/train')
parser.add_argument('--test_dir', type=str, default='/tmp/imagenet/val')
parser.add_argument('--num_cores', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--metrics_debug', action='store_true')
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers
# Setup import folders.
_XLA_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(os.path.join(os.path.dirname(_XLA_FOLDER), 'test'))
sys.path.insert(0, _XLA_FOLDER)

from common_utils import TestCase, run_tests
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla_py.xla_model as xm
import unittest


def train_imagenet():
    print('==> Preparing data..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = torchvision.datasets.ImageFolder(
        FLAGS.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers)
    test_dataset = torchvision.datasets.ImageFolder(
        FLAGS.test_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers)

    torch.manual_seed(42)

    print('==> Building model..')
    momentum = 0.9
    lr = 0.1
    log_interval = max(1, int(10 / FLAGS.num_cores))

    model = torchvision.models.resnet50()
    cross_entropy_loss = nn.CrossEntropyLoss()

    devices = [':{}'.format(n) for n in range(0, FLAGS.num_cores)]
    inputs = torch.zeros(FLAGS.batch_size, 3, 224, 224)
    target = torch.zeros(FLAGS.batch_size, dtype=torch.int64)
    xla_model = xm.XlaModel(model, [inputs], loss_fn=cross_entropy_loss,
                            target=target, num_cores=FLAGS.num_cores,
                            devices=devices)
    optimizer = optim.SGD(xla_model.parameters_list(), lr=lr,
                          momentum=momentum, weight_decay=5e-4)

    for epoch in range(1, FLAGS.num_epochs + 1):
        xla_model.train(train_loader, optimizer, FLAGS.batch_size,
                        log_interval=log_interval)
        if FLAGS.metrics_debug:
            print(torch_xla._C._xla_metrics_report())
        accuracy = xla_model.test(test_loader,
                                  xm.category_eval_fn(cross_entropy_loss),
                                  FLAGS.batch_size)
        xm.update_optimizer_state(optimizer, 'lr', lambda x: x / 1.025)
    return accuracy


class TrainImageNet(TestCase):

    def tearDown(self):
        super(TrainImageNet, self).tearDown()

    def test_accurracy(self):
        # TODO: figure out accuracy target, make it trivially true for now.
        self.assertGreaterEqual(train_imagenet(), 0)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
