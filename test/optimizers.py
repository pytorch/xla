import test_utils
from torch.optim import SGD

MIN_DELTA_TO_UPDATE_LR = 0.000001
NUM_WARMUP_EPOCHS = 1


class WarmupAndExponentialDecaySGD(SGD):
    """SGD with a step-based warmup period and epoch-based exponential decay.

    Args:
        TODO(zcain): fill in args here.
    """

    def __init__(self, params, lr=None, momentum=0, dampening=0, weight_decay=0,
                 nesterov=False, num_steps_per_epoch=None,
                 divide_every_n_epochs=20, divisor=0.2):
      self.num_steps_per_epoch = num_steps_per_epoch
      self.divide_every_n_epochs = divide_every_n_epochs
      self.divisor = divisor
      self.finished = False
      self.max_lr = lr
      self.previous_lr = -1
      super(WarmupAndExponentialDecaySGD, self).__init__(
              params, lr=lr, momentum=momentum, dampening=dampening,
              weight_decay=weight_decay, nesterov=nesterov)


    def get_lr(self, epoch, step):
      if epoch <= NUM_WARMUP_EPOCHS:
        # Warmup epoch. Ramp up learning rate from 0.0 to self.max_lr using
        # a linear slope.
        assert step is not None
        return min(self.max_lr,
                   self.max_lr * ((step + 1.0) / self.num_steps_per_epoch))

      # Normal epoch. Use an exponential decay determined by init params.
      # NOTE: We assume 1-indexed epochs here.
      return self.max_lr * (
          self.divisor ** ((epoch - 1) // self.divide_every_n_epochs))


    def step(self, epoch=None, step=None, summary_writer=None):
      if epoch is None or step is None:
        raise ValueError('epoch and step are required as keyword args.')
      current_lr = self.get_lr(epoch, step)

      # Update SGD's learning rate before calling optimizer.step().
      # Outside of warmup epochs, we use the same learning rate for every step
      # in an epoch. Don't bother updating learning rate if it hasn't changed.
      if abs(current_lr - self.previous_lr) > MIN_DELTA_TO_UPDATE_LR:
        for param_group in self.param_groups:
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

      # Allow SGD to take its normal step using the modified learning rate.
      super(WarmupAndExponentialDecaySGD, self).step()

