import jax
import torch
import torch_xla2
from torch_xla2.interop import JittableModule

from transformers.modeling_outputs import BaseModelOutputWithPooling

from jax.tree_util import register_pytree_node

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
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

prompt = "a photograph of an astronaut riding a horse"
# image = pipe(prompt).images[0]


env = torch_xla2.default_env()

def move_scheduler(scheduler):
  for k, v in scheduler.__dict__.items():
    if isinstance(v, torch.Tensor):
      setattr(scheduler, k, v.to('jax'))


with env:
  pipe.to('jax:1')
  #import pdb; pdb.set_trace()
  move_scheduler(pipe.scheduler)
  pipe.unet = JittableModule(pipe.unet, extra_jit_args={'static_argnames': ('return_dict',)})
  pipe.text_encoder = JittableModule(pipe.text_encoder)
  image = pipe(prompt).images[0]
  image.save(f"astronaut_rides_horse.png")



