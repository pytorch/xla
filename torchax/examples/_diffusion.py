import functools

import torch
from time import time
from diffusers import DiffusionPipeline
from torch.utils import _pytree as pytree

import torchax
import torchax.functions
from torchax.extra import torch_view, jax_view

import jax
import torch.func


class CompiledModule:

  def __init__(self, model):
    weights = model.state_dict()
    weights.update(model.named_parameters())
    self._weights = pytree.tree_map_only(torch.Tensor,
                                         torchax.tensor.move_to_device, weights)
    self._model = model

    self._func_jitted_torch = None  #torch_view(func_mod_jitted)

  def _maybe_move_tensor(self, tensor):
    if isinstance(
        tensor, torch.Tensor) and not isinstance(tensor, torchax.tensor.Tensor):
      return torchax.tensor.move_to_device(tensor)
    return tensor

  def _make_jitted(self, args, kwargs):
    static = []
    for i, a in enumerate(args):
      if not isinstance(a, torch.Tensor):
        static.append(i + 1)  # weight is 0
    static_argnames = []
    for k, v in kwargs.items():
      if not isinstance(v, torch.Tensor):
        static_argnames.append(k)

    def f(weights, *args, **kwargs):
      weights, args, kwargs = torchax.tensor.wrap((weights, args, kwargs))
      with torchax.functions.XLAFunctionMode(), torchax.tensor.XLADispatchMode(
      ):
        res = torch.func.functional_call(self._model, weights, args, kwargs)
        if isinstance(res, tuple) and len(res) == 1:
          res = res[0]
      return torchax.tensor.unwrap(res)

    fjit = jax.jit(f, static_argnames=tuple(static_argnames))
    return torch_view(fjit)

  def forward(self, *args, **kwargs):
    (args, kwargs) = pytree.tree_map(self._maybe_move_tensor, (args, kwargs))
    if self._func_jitted_torch is None:
      self._func_jitted_torch = self._make_jitted(args, kwargs)
    return self._func_jitted_torch(self._weights, *args, **kwargs)

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def __getattr__(self, key):
    return getattr(self._model, key)


def compile_pipe(pipe):
  pipe.text_encoder = CompiledModule(pipe.text_encoder)
  pipe.text_encoder_2 = CompiledModule(pipe.text_encoder_2)
  pipe.unet = CompiledModule(pipe.unet)
  pipe.vae = CompiledModule(pipe.vae)


def main():
  pipe = DiffusionPipeline.from_pretrained(
      # "stabilityai/stable-diffusion-xl-base-0.9",
      "stabilityai/stable-diffusion-xl-base-1.0",
      use_safetensors=True,
  )
  compile_pipe(pipe)

  global_bs = 10
  inference_steps = 20
  resol = 1024
  prompts = ["a photo of an astronaut riding a horse on mars"] * global_bs
  print(
      f'global batch size {global_bs}',
      f'inference steps {inference_steps}',
      f'Image resolution {resol}',
      flush=True)

  iters = 5
  for i in range(iters):
    prompt = prompts
    # print('per device prompts len',len(prompt))
    # prompt = prompts[rank]
    start = time()
    image = pipe(
        prompt, num_inference_steps=inference_steps, height=resol,
        width=resol).images[0]
    print(f'Step {i} inference time {time()-start} sec', flush=True)


if __name__ == '__main__':
  main()
