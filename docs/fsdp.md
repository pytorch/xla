## Fully Sharded Data Parallel (FSDP) in PyTorch XLA

Fully Sharded Data Parallel (FSDP) in PyTorch XLA is a utility for sharding Module parameters across data-parallel workers.

Example usage:
```python3
import torch
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

model = FSDP(my_module)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
output = model(x, y)
loss = output.sum()
loss.backward()
optim.step()
```
It is also possible to shard individual layers separately and have an outer wrapper handle any leftover parameters.

Notes:
* The `XlaFullyShardedDataParallel` class supports both the ZeRO-2 optimizer (sharding gradients and optimizer states) and the ZeRO-3 optimizer (sharding parameters, gradients, and optimizer states) in https://arxiv.org/abs/1910.02054.
  * The ZeRO-3 optimizer should be implemented via nested FSDP with `reshard_after_forward=True`. See `test/test_train_mp_mnist_fsdp_with_ckpt.py` and `test/test_train_mp_imagenet_fsdp.py` for an example. 
  * For large models that cannot fit into a single TPU memory or the host CPU memory, one should interleave submodule construction with inner FSDP wrapping. See [`FSDPViTModel`](https://github.com/ronghanghu/vit_10b_fsdp_example/blob/master/run_vit_training.py) for an example.
* a simple wrapper `checkpoint_module` is provided (based on `torch_xla.utils.checkpoint.checkpoint` from https://github.com/pytorch/xla/pull/3524) to perform [gradient checkpointing](https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs) over a given `nn.Module` instance. See `test/test_train_mp_mnist_fsdp_with_ckpt.py` and `test/test_train_mp_imagenet_fsdp.py` for an example.
* Auto-wrapping submodules: instead of manually nested FSDP wrapping, one can also specify an `auto_wrap_policy` argument to automatically wrap the submodules with inner FSDP. `size_based_auto_wrap_policy` in `torch_xla.distributed.fsdp.wrap` is an example of `auto_wrap_policy` callable, this policy wraps layers with the number of parameters larger than 100M. `transformer_auto_wrap_policy` in `torch_xla.distributed.fsdp.wrap` is an example of `auto_wrap_policy` callable for transformer-like model architectures.

For example, to automatically wrap all `torch.nn.Conv2d` submodules with inner FSDP, one can use:
```python3
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={torch.nn.Conv2d})
```

Additionally, one can also specify an `auto_wrapper_callable` argument to use a custom callable wrapper for the submodules (the default wrapper is just the `XlaFullyShardedDataParallel` class itself). For example, one can use the following to apply gradient checkpointing (i.e. activation checkpointing/rematerialization) to each auto-wrapped submodule.
```python3
from torch_xla.distributed.fsdp import checkpoint_module
auto_wrapper_callable = lambda m, *args, **kwargs: XlaFullyShardedDataParallel(
    checkpoint_module(m), *args, **kwargs)
```
* When stepping the optimizer, directly call `optimizer.step` and do not call `xm.optimizer_step`. The latter reduces the gradient across ranks, which is not needed for FSDP (where the parameters are already sharded).
* When saving model and optimizer checkpoints during training, each training process needs to save its own checkpoint of the (sharded) model and optimizer state dicts (use `master_only=False` and set different paths for each rank in `xm.save`). When resuming, it needs to load the checkpoint for the corresponding rank.
* Please also save `model.get_shard_metadata()` along with `model.state_dict()` as follows and use `consolidate_sharded_model_checkpoints` to stitch the sharded model checkpoints together into a full model state dict. See `test/test_train_mp_mnist_fsdp_with_ckpt.py` for an example.
```python3
ckpt = {
    'model': model.state_dict(),
    'shard_metadata': model.get_shard_metadata(),
    'optimizer': optimizer.state_dict(),
}
ckpt_path = f'/tmp/rank-{xm.get_ordinal()}-of-{xm.xrt_world_size()}.pth'
xm.save(ckpt, ckpt_path, master_only=False)
```
* The checkpoint consolidation script can also be launched from the command line as follows.
```bash
# consolidate the saved checkpoints via command line tool
python3 -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts \
  --ckpt_prefix /path/to/your_sharded_checkpoint_files \
  --ckpt_suffix "_rank-*-of-*.pth"
```

The implementation of this class is largely inspired by and mostly follows the structure of `fairscale.nn.FullyShardedDataParallel` in https://fairscale.readthedocs.io/en/stable/api/nn/fsdp.html. One of the biggest differences from `fairscale.nn.FullyShardedDataParallel` is that in XLA we don't have explicit parameter storage, so here we resort to a different approach to free full parameters for ZeRO-3.

---

### Example training scripts on MNIST and ImageNet

* MNIST: [`test/test_train_mp_mnist_fsdp_with_ckpt.py`](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_fsdp_with_ckpt.py) (it also tests checkpoint consolidation)
* ImageNet: [`test/test_train_mp_imagenet_fsdp.py`](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_fsdp.py)

#### Installation
FSDP is available on PyTorch/XLA 1.12 release and newer nightly. Please refer to https://github.com/pytorch/xla#-available-images-and-wheels for installation guide.

#### Clone PyTorch/XLA repo
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
git clone --recursive https://github.com/pytorch/xla.git
cd ~/
```
#### Train MNIST on v3-8 TPU
It gets around 98.9 accuracy for 2 epochs:
```bash
python3 ~/pytorch/xla/test/test_train_mp_mnist_fsdp_with_ckpt.py \
  --batch_size 16 --drop_last --num_epochs 2 \
  --use_nested_fsdp --use_gradient_checkpointing
```
This script automatically tests checkpoint consolidation at the end. You can also manually consolidate the sharded checkpoints via
```bash
# consolidate the saved checkpoints via command line tool
python3 -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts \
  --ckpt_prefix /tmp/mnist-fsdp/final_ckpt \
  --ckpt_suffix "_rank-*-of-*.pth"
```

#### Train ImageNet with ResNet-50 on v3-8 TPU

It gets around 75.9 accuracy for 100 epochs; download [ImageNet-1k](https://github.com/pytorch/examples/tree/master/imagenet#requirements) to `/datasets/imagenet-1k`:
```bash
python3 ~/pytorch/xla/test/test_train_mp_imagenet_fsdp.py \
  --datadir /datasets/imagenet-1k --drop_last \
  --model resnet50 --test_set_batch_size 64 --eval_interval 10 \
  --lr 0.4 --batch_size 128 --num_warmup_epochs 5 --lr_scheduler_divide_every_n_epochs 30 --lr_scheduler_divisor 10 --num_epochs 100 \
  --use_nested_fsdp
```
You can also add ` --use_gradient_checkpointing` (which needs to be used along with `--use_nested_fsdp` or `--auto_wrap_policy`) to apply gradient checkpointing on the residual blocks.

---

### Example training scripts on TPU pod (with 10 billion parameters)

To train large models that cannot fit into a single TPU, one should apply auto-wrap or manually wrap the submodules with inner FSDP when building the entire model to implement the ZeRO-3 algorithm.

Please see https://github.com/ronghanghu/vit_10b_fsdp_example for an example of sharded training of a Vision Transformer (ViT) model using this XLA FSDP PR.
