import test_utils

FLAGS = test_utils.parse_common_options(
    datadir='/tmp/imagenet', batch_size=128, num_epochs=15,
    target_accuracy=0.0)

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
import torch_xla_py.xla_model as xm
import unittest


def _cross_entropy_loss_eval_fn(cross_entropy_loss):
    def eval_fn(output, target):
        loss = cross_entropy_loss(output, target).item()
        # Get the index of the max log-probability.
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        return loss, correct

    return eval_fn


def train_imagenet():
    print('==> Preparing data..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'train'),
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
        os.path.join(FLAGS.datadir, 'val'),
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
                                  _cross_entropy_loss_eval_fn(cross_entropy_loss),
                                  FLAGS.batch_size)
        xm.update_optimizer_state(optimizer, 'lr', lambda x: x / 1.025)
    return accuracy


class TrainImageNet(TestCase):

    def tearDown(self):
        super(TrainImageNet, self).tearDown()

    def test_accurracy(self):
        self.assertGreaterEqual(train_imagenet(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
