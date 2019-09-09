import test_utils
from torch.optim.lr_scheduler import _LRScheduler

MIN_DELTA_TO_UPDATE_LR = 0.000001
NUM_WARMUP_EPOCHS = 1


class WarmupAndExponentialDecayScheduler(_LRScheduler):
  """Update the learning rate of wrapped optimizer based on epoch and step.
  
  Args:
    optimizer: Instance of torch.optim.Optimizer. Learning rate will be changed.
    num_steps_per_epoch: int, the number of steps required to finish 1 epoch.
    divide_every_n_epochs: After this number of epochs, learning rate will be
        divided by the `divisor` param.
    divisor: The learning rate will be divided by this amount when
        epoch % divide_every_n_epochs == 0 (epoch 0 is excluded).
    summary_writer: Instance of `torch.utils.tensorboard.SummaryWriter`. If
        provided, learning rate will be logged during calls to step if step
        is called with write_to_summary=True. If summary_writer is None, then
        no logging happens.
  """
  def __init__(self, optimizer, num_steps_per_epoch, divide_every_n_epochs=20,
               divisor=5, summary_writer=None):
    self.num_steps_per_epoch = num_steps_per_epoch
    self.divide_every_n_epochs = divide_every_n_epochs
    self.divisor = divisor
    self.previous_lr = -1
    self.max_lr = optimizer.param_groups[0]['lr']
    self.summary_writer = summary_writer
    super(WarmupAndExponentialDecayScheduler, self).__init__(optimizer)


  def _epoch(self):
    return self._step_count // self.num_steps_per_epoch


  def get_lr(self):
    epoch = self._epoch()
    lr = 0.0

    if epoch < NUM_WARMUP_EPOCHS:
      # Warmup epoch. Ramp up learning rate from 0.0 to self.max_lr using
      # a linear slope.
      lr = min(self.max_lr,
               self.max_lr * ((self._step_count + 1.0) / self.num_steps_per_epoch))
    else:
      # Normal epoch. Use an exponential decay determined by init params.
      lr = self.max_lr / (
          self.divisor ** (epoch // self.divide_every_n_epochs))

    # _LRScheduler expects a list of learning rates like this.
    return [lr for _ in self.base_lrs]


  def step(self, epoch=None):
    epoch = self._epoch()
    current_lr = self.get_lr()[0]

    # Add current learning rate to Tensorboard metrics. For warmup epochs,
    # log the learning rate at every step. For non-warmup epochs, log only
    # the first step since the entire epoch will use the same learning rate.
    if self.summary_writer:
      if self._epoch() < NUM_WARMUP_EPOCHS or self._step_count % self.num_steps_per_epoch == 0:
        test_utils.add_scalar_to_summary(self.summary_writer,
                                         'LearningRate',
                                         current_lr,
                                         self._step_count)

    # Outside of warmup epochs, we use the same learning rate for every step
    # in an epoch. Don't bother updating learning rate if it hasn't changed. 
    lr_delta = abs(current_lr - self.previous_lr)
    self.previous_lr = current_lr
    if lr_delta < MIN_DELTA_TO_UPDATE_LR:
      self._step_count += 1  # This normally happens in super().step().
      return

    super(WarmupAndExponentialDecayScheduler, self).step()


def wrap_optimizer_with_scheduler(optimizer, flags, num_steps_per_epoch=None, summary_writer=None):
  scheduler_type = getattr(flags, 'lr_scheduler_type', None)
  if not scheduler_type:
    return None

  if scheduler_type == 'WarmupAndExponentialDecayScheduler':
    scheduler_divisor = getattr(flags, 'lr_scheduler_divisor')
    scheduler_divide_every_n_epochs = getattr(
        flags, 'lr_scheduler_divide_every_n_epochs')
    if num_steps_per_epoch is None:
      raise ValueError(
          'WarmupAndExponentialDecayScheduler requires num_steps_per_epoch')
    return WarmupAndExponentialDecayScheduler(
            optimizer, num_steps_per_epoch,
            divide_every_n_epochs=scheduler_divide_every_n_epochs,
            divisor=scheduler_divisor, summary_writer=summary_writer)

