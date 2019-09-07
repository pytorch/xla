import test_utils
from torch.optim.lr_scheduler import _LRScheduler

MIN_DELTA_TO_UPDATE_LR = 0.000001
NUM_WARMUP_EPOCHS = 1


class WarmupAndExponentialDecayScheduler(_LRScheduler):
  """TODO(zcain): docstring."""
  def __init__(self, optimizer, num_steps_per_epoch, divide_every_n_epochs=20, divisor=5):
    self.num_steps_per_epoch = num_steps_per_epoch
    self.divide_every_n_epochs = divide_every_n_epochs
    self.divisor = divisor
    self.previous_lr = -1
    self.max_lr = optimizer.param_groups[0]['lr']
    self.optimizer = optimizer

  def get_lr(self, epoch, step):
    # Warmup epoch. Ramp up learning rate from 0.0 to self.max_lr using
    # a linear slope.
    if epoch <= NUM_WARMUP_EPOCHS:
      return min(self.max_lr,
                 self.max_lr * ((step + 1.0) / self.num_steps_per_epoch))

    # Normal epoch. Use an exponential decay determined by init params.
    # NOTE: We assume 1-indexed epochs here.
    return self.max_lr / (
        self.divisor ** ((epoch - 1) // self.divide_every_n_epochs))

  def step(self, epoch, step, summary_writer=None):
    current_lr = self.get_lr(epoch, step)

    # Outside of warmup epochs, we use the same learning rate for every step
    # in an epoch. Don't bother updating learning rate if it hasn't changed.
    if abs(current_lr - self.previous_lr) > MIN_DELTA_TO_UPDATE_LR:
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = current_lr
      self.previous_lr = current_lr

    # Add current learning rate to Tensorboard metrics. For warmup epochs,
    # log the learning rate at every step. For non-warmup epochs, log only
    # the first step since the entire epoch will use the same learning rate.
    if summary_writer:
      if epoch <= NUM_WARMUP_EPOCHS or step == 0:
        # NOTE: We assume 1-indexed epochs here.
        global_step = step + ((epoch - 1) * self.num_steps_per_epoch)
        test_utils.add_scalar_to_summary(summary_writer,
                                         'LearningRate',
                                         current_lr,
                                         global_step)
