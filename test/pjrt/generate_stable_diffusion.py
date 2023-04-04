import torch
import torch_xla.core.xla_model as xm
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline = pipeline.to(xm.xla_device())

prompt = "a cloud tpu engineer"
generator = torch.Generator().manual_seed(0)

def callback(step: int, timestamp: int, latents: torch.FloatTensor):
  print('step', step, timestamp)
  xm.mark_step()
  print('after mark')

image = pipeline(prompt, callback=lambda *args: xm.mark_step(), generator=generator).images[0]

image.save('output.jpg')
