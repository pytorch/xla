import os
import sys
import torch
import  numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
from time import time
from typing import Tuple
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline


def setup_model_parallel() -> Tuple[int, int]:
    # assuming model parallelism over the whole world size
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()

    # seed must be the same in all processes
    torch.manual_seed(1)
    device = xm.xla_device()
    xm.set_rng_state(1, device=device)
    return rank, world_size


def main(index):
    # starts server at port 9012 for profiling
    # (refer to https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
    server = xp.start_server(9012)
    device = xm.xla_device()
    rank, world_size = setup_model_parallel()
    print('rank, world_size', rank, world_size)

    # print only for xla:0 device
    if rank > 0:
        sys.stdout = open(os.devnull, "w")

    pipe = DiffusionPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-xl-base-0.9",
        "stabilityai/stable-diffusion-xl-base-1.0",
        use_safetensors=True,

        )
    pipe = pipe.to(device)

    global_bs = 288
    inference_steps = 20
    resol = 1024
    prompts = ["a photo of an astronaut riding a horse on mars"] * global_bs
    print(f'global batch size {global_bs}',
          f'inference steps {inference_steps}',
          f'Image resolution {resol}',
          flush=True
          )

    iters = 5
    for i in range(iters):

        prompt = prompts[rank::world_size]
        # print('per device prompts len',len(prompt))
        # prompt = prompts[rank]
        start = time()
        image = pipe(prompt,
                     num_inference_steps=inference_steps,
                     height=resol,
                     width=resol).images[0]
        print(f'Step {i} inference time {time()-start} sec', flush=True)


if __name__ == '__main__':
    xmp.spawn(main, args=())
