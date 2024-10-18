import time
import functools
import jax
import torch
import torch_xla2
from torch_xla2 import interop
from torch_xla2.interop import JittableModule

from transformers.modeling_outputs import BaseModelOutputWithPooling

from jax.tree_util import register_pytree_node
import jax

def base_model_output_with_pooling_flatten(v):
  return (v.last_hidden_state, v.pooler_output, v.hidden_states, v.attentions), None

def base_model_output_with_pooling_unflatten(aux_data, children):
  return BaseModelOutputWithPooling(*children)

register_pytree_node(
  BaseModelOutputWithPooling,
  base_model_output_with_pooling_flatten,
  base_model_output_with_pooling_unflatten
)


from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")

prompt = "a photograph of an astronaut riding a horse"
# image = pipe(prompt).images[0]


env = torch_xla2.default_env()
jax.config.update('jax_enable_x64', False)

def move_scheduler(scheduler):
  for k, v in scheduler.__dict__.items():
    if isinstance(v, torch.Tensor):
      setattr(scheduler, k, v.to('jax'))


with env:
  pipe.to('jax:1')
  move_scheduler(pipe.scheduler)
  pipe.unet = torch_xla2.compile(
    pipe.unet, torch_xla2.CompileOptions(
      jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
  )
  import pdb; pdb.set_trace()
  pipe.text_encoder = torch_xla2.compile(pipe.text_encoder)

  BS = 4
  prompt = [prompt] * BS 
  pipe.vae = torch_xla2.compile(
    pipe.vae, torch_xla2.CompileOptions(
      jax_jit_kwargs={'static_argnames': ('return_dict',)},
      methods_to_compile=['decode'],
    )
  )
  image = pipe(prompt).images[0]

  jax.profiler.start_trace('/tmp/sdxl')
  start = time.perf_counter()
  image = pipe(prompt, num_inference_steps=20).images[0]
  end = time.perf_counter()
  jax.profiler.stop_trace()
  print('Total time is ', end - start, 'bs = ', BS)
  image.save(f"astronaut_rides_horse.png")



