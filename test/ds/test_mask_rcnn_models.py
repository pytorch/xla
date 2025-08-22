import argparse
import os
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=2)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

import numpy as np
import unittest
import itertools
import random
from pathlib import Path
from typing import Tuple

import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

import torchvision
from coco_utils import ConvertCocoPolysToMask
from torchvision.datasets.coco import CocoDetection

# silence some spam
from pycocotools import coco

coco.print = lambda *args: None

CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
# Note[xw32]: need to copy the coco2017-minimal folder from torch benchmark to pytorch/xla/test/ds/data/.
DATA_DIR = os.path.join(CURRENT_DIR, "data", "coco2017-minimal")  # xw modified.
assert os.path.exists(
    DATA_DIR
), "Couldn't find coco2017 minimal data dir, please run install.py again."
COCO_DATA_KEY = "coco_2017_val_100"
COCO_DATA = {
    "coco_2017_val_100":
        ("coco/val2017", "coco/annotations/instances_val2017_100.json")
}

pd = torch._C._EnablePythonDispatcher()
xla_dev = xm.xla_device()


def _collate_fn(batch):
  return tuple(zip(*batch))


def _prefetch(loader, device):
  items = []
  for images, targets in loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    items.append((images, targets))
  return items


class Model():
  DEFAULT_TRAIN_BSIZE = 4
  DEFAULT_EVAL_BSIZE = 4
  NUM_OF_BATCHES = 1

  def __init__(self, test, device, batch_size=None, extra_args=[]):
    print('xw32 calling vision_maskrcnn.__init__')
    self.device = device
    if batch_size:
      self.batch_size = batch_size
    else:
      self.batch_size = self.DEFAULT_TRAIN_BSIZE if test == 'train' else self.DEFAULT_EVAL_BSIZE
    self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.
        COCO_V1).to(self.device)

    # setup optimizer
    # optimizer parameters copied from
    # https://github.com/pytorch/vision/blob/30f4d108319b0cd28ae5662947e300aad98c32e9/references/detection/train.py#L77
    lr = 0.02
    momentum = 0.9
    weight_decay = 1e-4
    params = [p for p in self.model.parameters() if p.requires_grad]
    self.optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    transforms = ConvertCocoPolysToMask()
    dataset = CocoDetection(
        root=os.path.join(DATA_DIR, COCO_DATA[COCO_DATA_KEY][0]),
        annFile=os.path.join(DATA_DIR, COCO_DATA[COCO_DATA_KEY][1]),
        transforms=transforms)
    sampler = torch.utils.data.SequentialSampler(dataset)

    self.data_loader = _prefetch(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=_collate_fn), self.device)

  def train(self):
    print('xw32 mask rcnn training starts.')
    self.model.train()
    for _batch_id, (images, targets) in zip(
        range(self.NUM_OF_BATCHES), self.data_loader):
      # images is a list. targets is a list of dict.
      # images[0] is a torch.tensor.
      # targets[0] is a dict with str key and torch.tensor value.
      # both images[i] and targets[i].value is on xla device.
      # sample_target_sample_key = list(targets[0].keys())[0]
      # print('next(self.model.parameters()).device=', next(self.model.parameters()).device) # prints xla:0.
      loss_dict = self.model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      self.optimizer.zero_grad()
      losses.backward()
      xm.optimizer_step(self.optimizer)

  def eval(self) -> Tuple[torch.Tensor]:
    self.model.eval()
    with torch.no_grad():
      for _batch_id, (images, _targets) in zip(
          range(self.NUM_OF_BATCHES), self.data_loader):
        out = self.model(images)
    out = list(map(lambda x: x.values(), out))
    return tuple(itertools.chain(*out))

class TestDynamicShapeMaskRCNN(unittest.TestCase):

  def test_mask_rcnn_training(self):
    model = Model('train', xla_dev)
    model.train()


if __name__ == '__main__':
  # assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
