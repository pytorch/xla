import test_utils
from torch.optim.lr_scheduler import _LRScheduler


class WarmupAndExponentialDecayScheduler(_LRScheduler):
  """Update the learning rate of wrapped optimizer based on epoch and step.
  
  Args:
    optimizer: Instance of torch.optim.Optimizer. Learning rate will be changed.
    num_steps_per_epoch: int, the number of steps required to finish 1 epoch.
    divide_every_n_epochs: After this number of epochs, learning rate will be
        divided by the `divisor` param.
    divisor: The learning rate will be divided by this amount when
        epoch % divide_every_n_epochs == 0 (epoch 0 is excluded).
    num_warmup_epochs: Learning rate will ramp up from 0 to max learning rate
        over this many epochs.
    min_delta_to_update_lr: If the new learning rate does not differ much from
        the learning rate of the previous step, don't bother updating the
        optimizer's learning rate.
    summary_writer: Instance of `torch.utils.tensorboard.SummaryWriter`. If
        provided, learning rate will be logged during calls to step if step
        is called with write_to_summary=True. If summary_writer is None, then
        no logging happens.
  """
  def __init__(self, optimizer, num_steps_per_epoch, divide_every_n_epochs=20,
               divisor=5, num_warmup_epochs=1, min_delta_to_update_lr=0.000001,
               summary_writer=None):
    self._num_steps_per_epoch = num_steps_per_epoch
    self._divide_every_n_epochs = divide_every_n_epochs
    self._divisor = divisor
    self._num_warmup_epochs = num_warmup_epochs
    self._min_delta_to_update_lr = min_delta_to_update_lr
    self._previous_lr = -1
    self._max_lr = optimizer.param_groups[0]['lr']
    self._summary_writer = summary_writer
    super(WarmupAndExponentialDecayScheduler, self).__init__(optimizer)

  def _epoch(self):
    return self._step_count // self._num_steps_per_epoch

  def get_lr(self):
    epoch = self._epoch()
    lr = 0.0

    if epoch < self._num_warmup_epochs:
      # Warmup epoch. Ramp up learning rate from 0.0 to self._max_lr using
      # a linear slope.
      num_warmup_steps = self._num_warmup_epochs * self._num_steps_per_epoch
      lr = min(self._max_lr,
               self._max_lr * ((self._step_count + 1.0) / num_warmup_steps))
    else:
      # Normal epoch. Use an exponential decay determined by init params.
      lr = self._max_lr / (
          self._divisor ** (epoch // self._divide_every_n_epochs))

    # _LRScheduler expects a list of learning rates like this.
    return [lr for _ in self.base_lrs]

  def step(self, epoch=None):
    current_lr = self.get_lr()[0]

    # Outside of warmup epochs, we use the same learning rate for every step
    # in an epoch. Don't bother updating learning rate if it hasn't changed. 
    if abs(current_lr - self._previous_lr) >= self._min_delta_to_update_lr:
      super(WarmupAndExponentialDecayScheduler, self).step()
      self._previous_lr = current_lr
    else:
      self._step_count += 1  # This normally happens in super().step().

    # Add current learning rate to Tensorboard metrics. For warmup epochs,
    # log the learning rate at every step. For non-warmup epochs, log only
    # the first step since the entire epoch will use the same learning rate.
    if self._summary_writer:
      if self._epoch() < self._num_warmup_epochs or (
          self._step_count % self._num_steps_per_epoch == 0):
        test_utils.add_scalar_to_summary(self._summary_writer,
                                         'LearningRate',
                                         self.optimizer.param_groups[0]['lr'],
                                         self._step_count)


def wrap_optimizer_with_scheduler(optimizer,
                                  scheduler_type=None,
                                  scheduler_divisor=None,
                                  scheduler_divide_every_n_epochs=None,
                                  num_steps_per_epoch=None,
                                  summary_writer=None):
  """Wraps an optimizer in a `torch.optim.lr_scheduler` object.

  Args:
    optimizer: Instance of `torch.optim.Optimizer`. Will be modified by the
        scheduler to overwrite the learning rate.
    scheduler_type: string, type of learning rate scheduler to use. If None,
        this method returns None.
    scheduler_divisor: int, required for WarmupAndExponentialDecayScheduler.
    scheduler_divide_every_n_epochs: int, required for
        WarmupAndExponentialDecayScheduler.
    num_steps_per_epoch: int, the number of steps that occur in each epoch.
        Required for WarmupAndExponentialDecayScheduler.
    summary_writer: Instance of `torch.utils.tensorboard.SummaryWriter` that
        will be passed into the scheduler to log learning rate during training.

  Raises:
    ValueError if the requested scheduler_type is unrecognized or if any
        required params are missing for the requested scheduler_type.
  """
  if not scheduler_type:
    return None

  if scheduler_type == 'WarmupAndExponentialDecayScheduler':
    if scheduler_divisor is None:
      raise ValueError('scheduler_divisor is required for '
                       'WarmupAndExponentialDecayScheduler.')
    if scheduler_divide_every_n_epochs is None:
      raise ValueError('scheduler_divide_every_n_epochs is required for '
                       'WarmupAndExponentialDecayScheduler.')
    if num_steps_per_epoch is None:
      raise ValueError('num_steps_per_epoch is required for '
                       'WarmupAndExponentialDecayScheduler.')
    return WarmupAndExponentialDecayScheduler(
            optimizer, num_steps_per_epoch,
            divide_every_n_epochs=scheduler_divide_every_n_epochs,
            divisor=scheduler_divisor, summary_writer=summary_writer)
  else:
    raise ValueError('Unknown scheduler_type: {}'.format(scheduler_type))

