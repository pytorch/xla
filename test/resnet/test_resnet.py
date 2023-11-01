import os
import args_parse
import random
from lars import create_optimizer_lars
from lars_utils import *
import resnet_model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
from torch_xla.experimental import pjrt
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import torch.distributed as dist
from torch_xla.amp import autocast
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch_xla.distributed.parallel_loader as pl
from torchvision import datasets
from itertools import islice
from folder2lmdb import ImageFolderLMDB

MODEL_OPTS = {
    '--train_batch_size': {
        'type': int,
    },
    '--eval_batch_size': {
        'type': int,
    },
    '--profile': {
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
    '--base_lr': {
        'type': float,
    },
    '--eeta': {
        'type': float,
    },
    '--end_lr': {
        'type': float,
    },
    '--epsilon': {
        'type': float,
    },
    '--weight_decay': {
        'type': float,
    },
    '--num_train_steps': {
        'type': int,
    },
    '--num_eval_steps': {
        'type': int,
    },
    '--warmup_epochs': {
        'type': int,
    },
    '--label_smoothing': {
        'type': float,
    },
    '--num_label_classes': {
        'type': int,
    },
    '--img_dim': {
        'type': int,
    },
    '--num_train_images': {
        'type': int,
    },
    '--num_eval_images': {
        'type': int,
    },
    '--amp': {
        'action': 'store_true',
    },
    '--train_only': {
        'action': 'store_true',
    },
    '--lmdb': {
        'action': 'store_true',
    },
    '--bn_bias_separately': {
        'action': 'store_true',
    },
    '--seed': {
        'type': int,
    },
}

DEFAULT_KWARGS = dict(
    train_batch_size=256,
    eval_batch_size=128,
    num_epochs=18,
    momentum=0.9,
    base_lr=17,
    target_accuracy=0.759,
    persistent_workers=True,
    num_label_classes=1000,
    prefetch_factor=32,
    loader_prefetch_size=16,
    device_prefetch_size=8,
    num_workers=16,
    host_to_device_transfer_threads=1,
    weight_decay=1e-4,
    seed=43,
    bn_bias_separately=True,
    img_dim=224,
    num_train_images=12000,
    num_eval_images=50000,
    train_only=False,
    label_smoothing=0.0,
    epsilon=1e-7,
    eeta=2e-4,   
    amp=True, 
)

FLAGS = args_parse.parse_common_options(
    datadir=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
)

# Set any args that were not explicitly given by the user.
default_value_dict = DEFAULT_KWARGS
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)

LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

class MultipleEpochsDataset(Dataset):
    def __init__(self, original_dataset, num_epochs):
        self.original_dataset = original_dataset
        self.num_epochs = num_epochs
        self.length = len(self.original_dataset) * self.num_epochs

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.original_dataset[index % len(self.original_dataset)]

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    
    # Check if the dataset is an instance of ImageFolderLMDB
    if isinstance(dataset, ImageFolderLMDB):
        dataset._ensure_opened()
    elif hasattr(dataset, 'datasets'):  # If it's a ConcatDataset or other composite dataset
        for sub_dataset in dataset.datasets:
            if isinstance(sub_dataset, ImageFolderLMDB):
                sub_dataset._ensure_opened()

def get_dataloaders():
  if FLAGS.datadir is None:
    train_loader = xu.SampleGenerator(
      data=(torch.rand(FLAGS.train_batch_size, 3, FLAGS.img_dim, FLAGS.img_dim),
              torch.randint(FLAGS.num_label_classes,(FLAGS.train_batch_size,), dtype=torch.int64)),
      sample_count=FLAGS.num_train_images)
    test_loader = xu.SampleGenerator(
      data=(torch.rand(FLAGS.eval_batch_size, 3, FLAGS.img_dim, FLAGS.img_dim),
              torch.randint(1000,(FLAGS.eval_batch_size,), dtype=torch.int64)),
      sample_count=FLAGS.num_eval_images // FLAGS.eval_batch_size // xm.xrt_world_size())
  else:
    train_dataset, test_dataset = get_lmdb_dataset()
    train_sampler, test_sampler = get_samplers(train_dataset, test_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.train_batch_size,
        sampler=train_sampler,
        drop_last=True,
        shuffle=False if train_sampler else True,
        num_workers=FLAGS.num_workers,
        persistent_workers=FLAGS.persistent_workers,
        worker_init_fn=worker_init_fn if FLAGS.lmdb else None,
        prefetch_factor=FLAGS.prefetch_factor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=FLAGS.eval_batch_size,
        sampler=test_sampler,
        drop_last=False,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        persistent_workers=FLAGS.persistent_workers,
        worker_init_fn=worker_init_fn if FLAGS.lmdb else None,
        prefetch_factor=FLAGS.prefetch_factor)
  train_device_loader = pl.MpDeviceLoader(
    train_loader,
    #train_loader if not FLAGS.train_only else islice(train_loader, 100000),
    xm.xla_device(),
    loader_prefetch_size=FLAGS.loader_prefetch_size,
    device_prefetch_size=FLAGS.device_prefetch_size,
    host_to_device_transfer_threads=FLAGS.host_to_device_transfer_threads)
  test_device_loader = pl.MpDeviceLoader(
    test_loader,
    xm.xla_device(),
    loader_prefetch_size=8,
    device_prefetch_size=4,
    host_to_device_transfer_threads=1)
  return train_device_loader, test_device_loader



def get_lmdb_dataset():
    resize_dim = FLAGS.img_dim
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    traindir = os.path.join(FLAGS.datadir, 'train.lmdb')
    train_dataset = ImageFolderLMDB(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(FLAGS.img_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.bfloat16),
            normalize,
        ]))
    train_dataset = MultipleEpochsDataset(train_dataset, 4)
    valdir = os.path.join(FLAGS.datadir, 'val.lmdb')
    test_dataset = ImageFolderLMDB(
        valdir,
        transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(FLAGS.img_dim),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.bfloat16),
            normalize,
        ]))
    eval_dataset_len = len(test_dataset)
    global_batch_size = FLAGS.eval_batch_size * xm.xrt_world_size()
    padding_needed = ((eval_dataset_len + global_batch_size - 1)// global_batch_size)  * global_batch_size - eval_dataset_len
    padding_needed = padding_needed//xm.xrt_world_size()
    padding_dataset = [(torch.zeros(3, FLAGS.img_dim, FLAGS.img_dim), torch.tensor([-1])) for _ in range(padding_needed)]
    test_dataset = ConcatDataset([test_dataset, padding_dataset])
    return train_dataset, test_dataset

def get_dataset():
  normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_dataset = torchvision.datasets.ImageFolder(
    os.path.join(FLAGS.datadir, 'train'),
    transforms.Compose([
        transforms.RandomResizedCrop(FLAGS.img_dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.bfloat16),
        normalize,
    ]))
  train_dataset = MultipleEpochsDataset(train_dataset, 4)
  resize_dim = max(FLAGS.img_dim, 256)
  test_dataset = torchvision.datasets.ImageFolder(
      os.path.join(FLAGS.datadir, 'val'),
      transforms.Compose([
          transforms.Resize(resize_dim),
          transforms.CenterCrop(FLAGS.img_dim),
          transforms.ToTensor(),
          transforms.ConvertImageDtype(torch.bfloat16),
          normalize,
      ]))
  # padding eval dataset
  # logic: calculate the per device shard size, pad the reminder with upperbound
  # and take only the per device shard size
  eval_dataset_len = len(test_dataset)
  global_batch_size = FLAGS.eval_batch_size * xm.xrt_world_size()
  padding_needed = ((eval_dataset_len + global_batch_size - 1)// global_batch_size)  * global_batch_size - eval_dataset_len
  padding_needed = padding_needed//xm.xrt_world_size()

  padding_dataset = [(torch.zeros(3, FLAGS.img_dim, FLAGS.img_dim), torch.tensor([-1])) for _ in range(padding_needed)]
  test_dataset = ConcatDataset([test_dataset, padding_dataset])
  return train_dataset,test_dataset

def get_samplers(train_dataset, test_dataset):
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
  return train_sampler, test_sampler

def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
    device,
    step,
    loss.item(),
    tracker.rate(),
    tracker.global_rate(),
    epoch,
    summary_writer=writer)
  #xm.master_print(f'loss: {loss.item()}')

def train_imagenet():
  torch.manual_seed(FLAGS.seed)
  device = xm.xla_device()
  model =  resnet_model.Resnet50(FLAGS.num_label_classes).to(device)
  train_device_loader, test_device_loader = get_dataloaders()

  # Initialization is nondeterministic with multiple threads in PjRt.
  # Synchronize model parameters across replicas manually.
  if pjrt.using_pjrt():
    pjrt.broadcast_master_param(model)

  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(FLAGS.logdir)

  xm.master_print(f'lr: {FLAGS.base_lr}')
  xm.master_print(f'eeta: {FLAGS.eeta}')
  xm.master_print(f'epsilon: {FLAGS.epsilon}')
  xm.master_print(f'momentum: {FLAGS.momentum}')
  xm.master_print(f'weight_decay: {FLAGS.weight_decay}')

  optimizer = create_optimizer_lars(model = model,
                                    lr = FLAGS.base_lr,
                                    eeta = FLAGS.eeta,
                                    epsilon=FLAGS.epsilon,
                                    momentum=FLAGS.momentum,
                                    weight_decay=FLAGS.weight_decay,
                                    bn_bias_separately=FLAGS.bn_bias_separately)

  steps_per_epoch = len(train_device_loader)
  print(f'steps_per_epoch_per_device: {steps_per_epoch}')

  lr_scheduler = PolynomialWarmup(optimizer,
                                  decay_steps=steps_per_epoch * FLAGS.num_epochs,
                                  warmup_steps=steps_per_epoch * FLAGS.warmup_epochs,
                                  end_lr=0.0,
                                  power=2.0,
                                  last_epoch=-1)

  loss_fn = LabelSmoothLoss(FLAGS.label_smoothing)

  if FLAGS.profile:
    server = xp.start_server(FLAGS.profiler_port)



  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):
      with xp.StepTrace('train_imagenet'):
      #  with xp.Trace('build_graph'):
        optimizer.zero_grad()
        if FLAGS.amp:
          with autocast(xm.xla_device()):
            output = model(data)
            loss = loss_fn(output, target)
          loss.backward()
          xm.optimizer_step(optimizer)
        else:
          output = model(data)
          loss = loss_fn(output, target)
          loss.backward()
          xm.optimizer_step(optimizer)
        if lr_scheduler:
          lr_scheduler.step()
        tracker.add(FLAGS.train_batch_size)
        if (step+1) % FLAGS.log_steps == 0:
          xm.add_step_closure(
              _train_update, args=(device, step, loss, tracker, epoch, writer))

  def test_loop_fn(loader, epoch):
    model.eval()
    total_samples, correct = 0, 0
    for step, (data, target) in enumerate(loader):
      with xp.Trace('eval_imagenet'):
        output = model(data)
        # Extrace un padded rows
        non_pad_indices = target!=-1
        output = output[non_pad_indices]
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target[non_pad_indices].view_as(pred)).sum()
        total_samples += output.size()[0]
    accuracy = 0.0
    if total_samples != 0:
      accuracy = 100.0 * correct / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy.item()

  accuracy, max_accuracy = 0.0, 0.0
  for epoch in range(1, (FLAGS.num_epochs + 1+3)//4):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader, epoch)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
    if not FLAGS.train_only and epoch: # run eval every 4 epochs
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

def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_imagenet()
  if accuracy < FLAGS.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  FLAGS.target_accuracy))

if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
