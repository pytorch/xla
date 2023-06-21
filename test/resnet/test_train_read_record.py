import imagenet_input
import time
import args_parse
from lars import create_optimizer_lars
from lars_utils import *
import resnet_model
import torch_xla
import torch
import os
import pprint
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
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.experimental import pjrt
import torch_xla.test.test_utils as test_utils
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torch_xla.amp import autocast


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
    '--warmup_steps': {
        'type': int,
    },
    '--label_smoothing': {
        'type': float,
    },
    '--num_classes': {
        'type': int,
    },
    '--amp': {
        'action': 'store_true',
    },
    '--iterations_per_loop': {
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
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
)


DEFAULT_KWARGS = dict(
    train_batch_size=256,
    eval_batch_size=256,
    num_epochs=44,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
    persistent_workers=True,
    prefetch_factor=32,
    loader_prefetch_size=16,
    device_prefetch_size=2,
    num_workers=8,
    host_to_device_transfer_threads=8,
    num_train_steps=10000,
    num_eval_steps=6,
    base_lr=17,
    eeta=1e-3,
    epsilon=0.0,
    weight_decay=2e-4,
    warmup_steps=781,
    end_lr=0.0,
    label_smoothing=0.1,
    num_classes = 1000,
    iterations_per_loop=3000,

)

default_value_dict = DEFAULT_KWARGS
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)

def _train_update(device, step, loss, tracker, epoch, writer):
  #avg_loss = xm.mesh_reduce('train_loss', loss.item(), np.mean)
  #xm.master_print(f'avg_loss: {avg_loss}')
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


#model_g = xmp.MpModelWrapper(resnet_model.Resnet50())

def train():
    torch.manual_seed(42)
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(None)

    imagenet_train = imagenet_input.get_input_fn(  # pylint: disable=g-complex-comprehension
          '/mnt/disks/persist/imagenet-1024',
          #None,
          True,
          tf.float32,
          224,
          True,
          cache_decoded_image=True)({'batch_size': 256, 'dataset_index': (xm.get_ordinal()), 'dataset_num_shards':(xm.xrt_world_size())})
    imagenet_eval = imagenet_input.get_input_fn(  # pylint: disable=g-complex-comprehension
          '/mnt/disks/persist/imagenet-1024',
          False,
          tf.float32,
          224,
          True,
          cache_decoded_image=True)({'batch_size': 256, 'dataset_index': (xm.get_ordinal()), 'dataset_num_shards':(xm.xrt_world_size())})
    #print(f'ordinal no: {xm.get_ordinal()//4}')
    device = xm.xla_device()
    train_loader=pl.MpDeviceLoader(imagenet_train,device,loader_prefetch_size=64,
      device_prefetch_size=32,
      host_to_device_transfer_threads=1)
    eval_loader = pl.MpDeviceLoader(imagenet_eval,device,loader_prefetch_size=8,
          device_prefetch_size=4,
          host_to_device_transfer_threads=1)

    model = resnet_model.Resnet50().to(device)
    pjrt.broadcast_master_param(model)
    optimizer = create_optimizer_lars(model = model,
                                        lr =17.0,
                                        eeta = 1e-3,
                                        epsilon=0.0,
                                        momentum=0.9,
                                        weight_decay=2e-4,
                                        bn_bias_separately=True)
    lr_scheduler = PolynomialWarmup(optimizer, decay_steps=6875,
                                    warmup_steps=782,
                                    end_lr=0.0, power=2.0, last_epoch=-1)

    loss_fn = LabelSmoothLoss(FLAGS.label_smoothing)

    server = xp.start_server(9229)
    tracker = xm.RateTracker()
    MEAN_RGB = xm.send_cpu_data_to_device(torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(3, 1, 1), device)
    STDDEV_RGB = xm.send_cpu_data_to_device(torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(3, 1, 1), device)
    def eval_loop(eval_loader1, num_eval_steps=10):
        
        model.eval()
        eval_iter = iter(eval_loader)
        total_samples, total_correct = 0, 0
        for step in range(num_eval_steps):
          data, target = next(eval_iter)
          num_samples, num_correct = eval_step(data, target) 
          total_samples += num_samples
          total_correct += num_correct
          if (step+1) % FLAGS.log_steps == 0:
                  xm.add_step_closure(
                    test_utils.print_test_update, args=(device, None, 0, step))
        accuracy = 100.0 * total_correct/total_samples
        accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
        xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
          0, test_utils.now(), accuracy))
        test_utils.write_to_summary(
              writer,
              0,
              dict_to_write={'Accuracy/test': accuracy},
              write_xla_metrics=True)

    def eval_step(data,target):
        #data = data.view(224,224,3,-1)
        #data = data.permute(3,2,0,1)
        #data = data.to(torch.bfloat16)
        data -= MEAN_RGB
        data /= STDDEV_RGB
        total_samples, correct = 0, 0
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
        total_samples += data.size()[0]
        return total_samples, correct
    def train_loop(train_loader, iterations_per_loop=FLAGS.num_train_steps,epoch=1):
        model.train()
        train_iter = iter(train_loader)
        for step in range(iterations_per_loop):
            with xp.StepTrace('train_loop',step_num=step):
                (data,target) = next(train_iter)
                train_step(data,target, step, epoch)

    def train_step(data, target, step,epoch=1):
        with xp.Trace('train_step'):
            #with xp.Trace('transformation_time'):
                #data = data.view(224, 224, -1, 3)
                #data = data.permute(2,3,0,1)
                #data = data.to(torch.bfloat16)
                #data -= MEAN_RGB
                #data /= STDDEV_RGB
            optimizer.zero_grad()
            with xp.Trace('tracing_time'):
              with autocast(xm.xla_device()):
                  output = model(data)
                  loss = loss_fn(output, target)
              loss.backward()
              xm.optimizer_step(optimizer)
              with xp.Trace('scheduler'):
                if lr_scheduler:
                    lr_scheduler.step()
            tracker.add(256)
            if (step+1)%FLAGS.log_steps == 0:
                #xm.mark_step()
                
                xm.add_step_closure(
                  _train_update, args=(device, step, loss, tracker, epoch, writer))


    def train_and_eval():
        epochs =  1 #FLAGS.num_train_steps // FLAGS.iterations_per_loop
        for epoch in range(1,epochs+1):
            train_loop(train_loader, 3600, epoch= epoch)
            #eval_loop(None, 7)
            if FLAGS.metrics_debug:
              xm.master_print(met.metrics_report())

    train_and_eval()
    return 0
    

def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train()


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
