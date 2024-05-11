from train_resnet_base import TrainResNetBase
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm

class TrainResNetXLADDP(TrainResNetBase):
    def run_optimizer(self):
        xm.optimizer_step(self.optimizer)

def _mp_fn(index):
    xla_ddp = TrainResNetXLADDP()
    xla_ddp.start_training()      

if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=())