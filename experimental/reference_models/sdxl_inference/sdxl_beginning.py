import torch
from diffusers import StableDiffusionPipeline

import torch_xla2
env = torch_xla2.default_env()

# this is now contains torhc.Tensor
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")

with env:
  pipe.to('jax')
  prompt = "a photograph of an astronaut riding a horse"
  image = pipe(prompt, num_inference_steps=10).images[0]
  image.save(f"astronaut_rides_horse_orig.png")
