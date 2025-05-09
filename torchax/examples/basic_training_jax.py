"""
This is the script from this tutorial:
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""

import functools
from torchax import train, interop
import torch
from torch.utils import _pytree as pytree
import torchvision
import torchvision.transforms as transforms
import torchax
import torchax.interop
import jax
import optax
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

env = torchax.enable_globally()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST(
    './data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST(
    './data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(
    training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):

  def __init__(self):
    super(GarmentClassifier, self).__init__()
    self.fc1 = nn.Linear(28 * 28, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


model = GarmentClassifier()
loss_fn = torch.nn.CrossEntropyLoss()

jax_optimizer = optax.adam(0.01)

model.to('jax')  # move the model to jax device
model_jittable = interop.JittableModule(model)
weights = model_jittable.params  # these are trainable parameters
buffers = model_jittable.buffers  # these are non-trainable parameters

opt_state = interop.call_jax(jax_optimizer.init, weights)
model_fn = functools.partial(model_jittable.functional_call, 'forward')

train_step = train.make_train_step(model_fn, loss_fn, jax_optimizer)

train_step = interop.jax_jit(
    train_step, kwargs_for_jax_jit={'donate_argnums': (0, 2)})

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_inputs = torch.rand(4, 28, 28).to('jax')
dummy_outputs = torch.rand(4, 10).to('jax')
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 5, 3, 7]).to('jax')

# test train_step


def train_one_epoch(weights, buffers, opt_state, epoch_index, tb_writer):
  running_loss = 0.
  last_loss = 0.

  # Here, we use enumerate(training_loader) instead of
  # iter(training_loader) so that we can track the batch
  # index and do some intra-epoch reporting
  for i, data in enumerate(training_loader):
    inputs, labels = data

    inputs = inputs.to('jax')
    labels = labels.to('jax')

    loss, weights, opt_state = train_step(weights, buffers, opt_state, inputs,
                                          labels)

    # Gather data and report
    running_loss += loss.item()
    if i % 1000 == 999:
      last_loss = running_loss / 1000  # loss per batch
      print('  batch {} loss: {}'.format(i + 1, last_loss))
      tb_x = epoch_index * len(training_loader) + i + 1
      tb_writer.add_scalar('Loss/train', last_loss, tb_x)
      running_loss = 0.

  return last_loss, weights, opt_state


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0
EPOCHS = 2
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
  print('EPOCH {}:'.format(epoch_number + 1))

  avg_loss, weights, opt_state = train_one_epoch(weights, buffers, opt_state,
                                                 epoch_number, writer)
  print(avg_loss)
