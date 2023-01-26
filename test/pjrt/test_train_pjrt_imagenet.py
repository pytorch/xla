import args_parse
import time

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
import pprint
import schedulers
import numpy as np
import sys
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
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.profiler as xp
from torch_xla.experimental import pjrt
import torch_xla.test.test_utils as test_utils

DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=128,
    num_epochs=18,
    momentum=0.9,
    lr=0.01,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
    'resnet50':
        dict(
            DEFAULT_KWARGS, **{
                'lr': 0.1,
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train_imagenet(flags):
  print('==> Preparing data..')

  device = xm.xla_device()
  #torch.multiprocessing.set_start_method('spawn')
  img_dim = get_model_property('img_dim')
  if flags.fake_data:
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    train_loader = xu.SampleGenerator(
        data=(torch.rand(flags.batch_size, 3, img_dim, img_dim),
              torch.randint(0,1000,(flags.batch_size,), dtype=torch.int64)),
        sample_count=train_dataset_len // flags.batch_size //
        xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.rand(flags.test_set_batch_size, 3, img_dim, img_dim),
              torch.randint(0,1000,(flags.test_set_batch_size,), dtype=torch.int64)),
        sample_count=50000 // flags.batch_size // xm.xrt_world_size())
  else:
    print("datadir working")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(flags.datadir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(img_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset_len = len(train_dataset.imgs)
    resize_dim = max(img_dim, 256)
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(flags.datadir, 'val'),
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
        batch_size=flags.batch_size,
        sampler=train_sampler,
        drop_last=flags.drop_last,
        shuffle=False if train_sampler else True,
        pin_memory=flags.pin_memory,
        persistent_workers=flags.persistent_workers,
        prefetch_factor=flags.prefetch_factor,
        num_workers=flags.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=flags.batch_size,
        sampler=test_sampler,
        drop_last=flags.drop_last,
        shuffle=False,
        pin_memory=flags.pin_memory,
        persistent_workers=flags.persistent_workers,
        prefetch_factor=flags.prefetch_factor,
        num_workers=flags.num_workers)


  model = get_model_property('model_fn')()
  model = model.to(device)
  pjrt.broadcast_master_param(model)
  server = xp.start_server(9229)
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
  optimizer = optim.SGD(
      model.parameters(),
      lr=flags.lr,
      momentum=flags.momentum,
      weight_decay=1e-4)
  num_training_steps_per_epoch = train_dataset_len // (
      flags.batch_size * xm.xrt_world_size())
  lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
      optimizer,
      scheduler_type=getattr(flags, 'lr_scheduler_type', None),
      scheduler_divisor=getattr(flags, 'lr_scheduler_divisor', None),
      scheduler_divide_every_n_epochs=getattr(
          flags, 'lr_scheduler_divide_every_n_epochs', None),
      num_steps_per_epoch=num_training_steps_per_epoch,
      summary_writer=writer)
  loss_fn = nn.CrossEntropyLoss()
  data_time = AverageMeter('Data_loading', ':6.3f')
  batch_time = AverageMeter('Batch_process_time', ':6.3f')
  
  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    #end = time.time()
    for step, (data, target) in enumerate(loader):
      #data_time.update(time.time()-end)
      with xp.StepTrace('train_loop', step_num = step):
        with xp.Trace('forward_pass'):
          #data = xm.send_cpu_data_to_device(data, device)
          #target = xm.send_cpu_data_to_device(target, device)
          optimizer.zero_grad()
          output = model(data)
          loss = loss_fn(output, target)
          loss.backward()
        xm.optimizer_step(optimizer)
        tracker.add(flags.batch_size)
        if lr_scheduler:
          lr_scheduler.step()
        if step % flags.log_steps == 0:
            xm.add_step_closure(
                _train_update, args=(device, step, loss, tracker, epoch, writer))
      '''
      with xp.StepTrace('wait_device_ops', step_num=step):
        #data = xm.send_cpu_data_to_device(data,device)
        #target = xm.send_cpu_data_to_device(target, device)
        xm.wait_device_ops(devices=None)
      '''
      #batch_time.update(time.time() - end)
      #end = time.time()
      #if step%flags.log_steps==0:
      #  xm.master_print(f'End of step: {step}')

    #xm.master_print(data_time.__str__() + ' seconds.')
    #xm.master_print(batch_time.__str__() + ' seconds.')

  def test_loop_fn(loader, epoch):
    total_samples, correct = 0, 0
    model.eval()
    for step, (data, target) in enumerate(loader):
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum()
      total_samples += data.size()[0]
      if step % flags.log_steps == 0:
        xm.add_step_closure(
            test_utils.print_test_update, args=(device, None, epoch, step))
    accuracy = 100.0 * correct.item() / total_samples
    # accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  train_device_loader = pl.MpDeviceLoader(train_loader, device, loader_prefetch_size= flags.loader_prefetch_size, device_prefetch_size=flags.device_prefetch_size, cpu_to_device_transfer_threads=4)
  test_device_loader = pl.MpDeviceLoader(test_loader, device, loader_prefetch_size = flags.loader_prefetch_size, device_prefetch_size=flags.device_prefetch_size, cpu_to_device_transfer_threads=4)
  accuracy, max_accuracy = 0.0, 0.0
  for epoch in range(1, flags.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader, epoch)
    
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
    if not flags.test_only_at_end or epoch == flags.num_epochs:
      accuracy = test_loop_fn(test_device_loader, epoch)
      xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
          epoch, test_utils.now(), accuracy))
      max_accuracy = max(accuracy, max_accuracy)
      test_utils.write_to_summary(
          writer,
          epoch,
          dict_to_write={'Accuracy/test': accuracy},
          write_xla_metrics=True)
    if flags.metrics_debug:
      xm.master_print(met.metrics_report())
  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy



if __name__ ==  '__main__':
  torch.set_default_tensor_type('torch.FloatTensor')

  results = pjrt._run_multiprocess(train_imagenet, FLAGS)
  print('Replica max_accuracy:', pprint.pformat(results))
  accuracy = np.mean(list(results.values()))
  print('Average max_accuracy:', accuracy)

  if accuracy < FLAGS.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  FLAGS.target_accuracy))
    sys.exit(21)
