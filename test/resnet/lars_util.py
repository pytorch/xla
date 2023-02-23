import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class LabelSmoothLoss(torch.nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

def create_lr_schedule(workers, warmup_epochs, decay_schedule, alpha=0.1):
    def lr_schedule(epoch):
        lr_adj = 1.
        if epoch < warmup_epochs:
            lr_adj = 1. / workers * (epoch * (workers - 1) / warmup_epochs + 1)
        else:
            decay_schedule.sort(reverse=True)
            for e in decay_schedule:
                if epoch >= e:
                    lr_adj *= alpha
        return lr_adj
    return lr_schedule



class PolynomialDecay(_LRScheduler):
    def __init__(self, optimizer, decay_steps, end_lr=0.0001, power=1.0, last_epoch=-1):
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        return [
            (base_lr - self.end_lr) * ((1 - min(self.last_epoch, self.decay_steps) /
                                        self.decay_steps) ** self.power) + self.end_lr
            for base_lr in self.base_lrs
        ]

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [self.last_epoch / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super().step(epoch)


class PolynomialWarmup(WarmupScheduler):
    def __init__(self, optimizer, decay_steps, warmup_steps=0, end_lr=0.0001, power=1.0, last_epoch=-1):
        base_scheduler = PolynomialDecay(
            optimizer, decay_steps - warmup_steps, end_lr=end_lr, power=power, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)
