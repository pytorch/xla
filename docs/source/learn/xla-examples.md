# Converting code to PyTorch XLA

General guidelines to modify your code:

-   Replace `cuda` with `torch_xla.device()`
-   Remove code that would access the XLA tensor values
-   Wrap data loader with MPDeviceLoader
-   Profile to further optimize the code

Remember: each case is unique so you might need to do something
different for each case.

## Example 1. Stable Diffusion inference in PyTorch Lightning on a Single TPU Device

To get a better understanding of the code changes needed to convert PyTorch code
that runs on GPUs to run on TPUs, let's look at the [inference
code](https://github.com/pytorch-tpu/stable-diffusion/blob/main/scripts/txt2img.py)
from a PyTorch implementation of the stable diffusion model. You can run the
script from the command line:

``` bash
    python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse"
```

To see a diff of the modifications explained below, see
[ldm/models/diffusion/ddim.py](https://github.com/pytorch-tpu/stable-diffusion/commit/57f398eb784387e244dc5fb78421aa5261abd1ef).
Let's go over them step by step. As in the general guidelines above,
start with changes related to `cuda` devices. This inference code is
written to run on GPUs and `cuda` can be found in multiple places. Start
making changes by removing `model.cuda()` from [this
line](https://github.com/pytorch-tpu/stable-diffusion/blob/978da4c625a712a01ee066d019a0b0d2319cd8b3/scripts/txt2img.py#L64),
and `precision_scope` from
[here](https://github.com/pytorch-tpu/stable-diffusion/blob/978da4c625a712a01ee066d019a0b0d2319cd8b3/scripts/txt2img.py#L290).
Additionally, replace the `cuda` device in [this
line](https://github.com/pytorch-tpu/stable-diffusion/blob/978da4c625a712a01ee066d019a0b0d2319cd8b3/scripts/txt2img.py#L248)
with the `xla` device similar to the code below:

``` python
    import torch_xla.core.xla_model as xm
    self.device = torch_xla.device()
```

Next, this particular configuration of the model is using
`FrozenCLIPEmbedder`, therefore we will modify this
[line](https://github.com/pytorch-tpu/stable-diffusion/blob/978da4c625a712a01ee066d019a0b0d2319cd8b3/ldm/modules/encoders/modules.py#L143)
as well. For simplicity we will directly define the `device` in this
tutorial, but you can pass the `device` value to the function as well.

Another place in the code that has cuda specific code is [DDIM scheduler](https://github.com/pytorch-tpu/stable-diffusion/blob/978da4c625a712a01ee066d019a0b0d2319cd8b3/ldm/models/diffusion/ddim.py#L12).
Add `import torch_xla.core.xla_model as xm` on top of the file then
replace
[these](https://github.com/pytorch-tpu/stable-diffusion/blob/978da4c625a712a01ee066d019a0b0d2319cd8b3/ldm/models/diffusion/ddim.py#L21-L22)
lines

``` python
if attr.device != torch.device("cuda"):
   attr = attr.to(torch.device("cuda"))
```

with

``` python
device = torch_xla.device()
attr = attr.to(torch.device(device))
```

Next, you can reduce device (TPU) and host (CPU) communication by
removing print statements, disabling progress bars, and reducing or
removing callbacks and logging. These operations require the device to
stop executing, falling back to the CPU, executing the
logging/callbacks, and then returning to the device. This can be a
significant performance bottleneck, especially on large models.

After making these changes, the code will run on TPUs. However, the
performance will not be optimized. This is because the XLA compiler tries to
build a single (huge) graph that wraps the number of inference steps (in
this case, 50) as there is no barrier inside the for loop. It is
difficult for the compiler to optimize the graph, and this leads to
significant performance degradation. As discussed above, breaking the
for loop with a call to `torch_xla.sync()` will result in a smaller
graph that is easier for the compiler to optimize. This allows
the compiler to reuse the graph from the previous step, which can
improve performance.

Now the
[code](https://github.com/pytorch-tpu/stable-diffusion/blob/ss-inference/scripts/txt2img.py)
is ready to run on TPUs in a reasonable time. More optimization and
analysis can be done by [capturing a
profile](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
and investigating further. However, this is not covered here.

Note: if you are running on v4-8 TPU, then you have 4 available XLA
(TPU) devices. Running the code as above will only use one XLA device.
In order to run on all 4 devices, use the `torch_xla.launch()` function.
We will discuss a `torch_xla.launch` in the next example.

## Example 2. HF Stable Diffusion Inference

Now, consider using [Stable Diffusion
Inference](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
in the HuggingFace diffusers library for both the SD-XL and 2.1 versions
of the model. You can find the changes described below in the [diffusers repo](https://github.com/pytorch-tpu/diffusers).
Clone the repo and run the inference script using the following command on your
TPU VM:

``` bash
(vm)$ git clone https://github.com/pytorch-tpu/diffusers.git
(vm)$ cd diffusers/examples/text_to_image/
(vm)$ python3 inference_tpu_single_device.py
```

## Running on a Single TPU device

This section describes how to update the
[text_to_image inference example](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image#inference)
to run on TPUs.

The original code uses Lora for inference, but this tutorial will not
use it. Instead, we will set the `model_id` argument to
`stabilityai/stable-diffusion-xl-base-0.9` when initializing the
pipeline. We will also use the default scheduler
(DPMSolverMultistepScheduler). However, similar changes can be made to
the other schedulers as well.

``` bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install . # pip install -e .

cd examples/text_to_image/
pip install -r requirements.txt
pip install invisible_watermark transformers accelerate safetensors
```

(If `accelerate` is not found, log out, log back in.)

Log in to HF and agree to the [sd-xl 0.9
license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)
on the model card. Next, go to
[account→settings→access](https://huggingface.co/settings/tokens) and generate a
new token. Copy the token and run the following command
with that specific token value on your vm

``` bash
(VM)$ huggingface-cli login --token _your_copied_token__
```

The [HuggingFace readme](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9#sd-xl-09-base-model-card)
provides PyTorch code that is written to run on
GPUs. To run it on TPUs, the first step is to change the CUDA device to
an XLA device. This can be done by replacing the line `pipe.to("cuda")`
with the following lines:

``` python
import torch_xla.core.xla_model as xm
device = torch_xla.device()
pipe.to(device)
```

The first time you run an inference with XLA, the compiler builds a graph of the
computations, and optimizes this graph for the specific hardware the code is
running on. Once the graph has been compiled, is can be reused for subsequent
calls, which will be much faster. For example, compilation time for stable
diffusion XL model inference from HuggingFace can take about an hour to compile,
whereas the actual inference may take only 5 seconds, depending on the batch
size. Likewise, a GPT-2 model can take about 10-15 mins to compile, after
which the training epoch time becomes much faster.

If you are running inference multiple times, you will start to see the
advantages of XLA after the graph is compiled. For example, if you
run inference on a list of 10 prompts, the first inference (maybe
two[^1]) may take a long time to compile, but the remaining inference
steps will be much faster. This is because XLA will reuse the graph that
it compiled for the first inference.

If you try to run the code without making any additional changes, you
will notice that the compilation time is very long (\>6 hours). This is
because the XLA compiler tries to build a single graph for all of the
scheduler steps at once similar to what we have discussed in the
previous example. To make the code run faster, we need to break the
graph up into smaller pieces with `torch_xla.sync()` and reuse them in the
next steps. This happens inside the `pipe.__call__`
[function](https://github.com/huggingface/diffusers/blob/2b1786735e27bc97f4d4699712292d5c463a7380/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L559)
in [these
lines](https://github.com/huggingface/diffusers/blob/2b1786735e27bc97f4d4699712292d5c463a7380/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L805-L839).
Disabling the progress bar, removing callbacks and adding
`torch_xla.sync()` at the end of the for loop speeds up the code
significantly. Changes are provided in this
[commit](https://github.com/huggingface/diffusers/compare/main...pytorch-tpu:diffusers:main).

Additionally, the `self.scheduler.step()` function, which by default
uses the `DPMSolverMultistepScheduler` scheduler, has a few issues that
are described in the [PyTorch XLA
caveats](https://pytorch.org/xla/release/2.0/index.html#known-performance-caveats).
The `.nonzero()` and `.item()` calls in this function send requests to
the CPU for tensor evaluation, which trigger device-host communication.
This is not desirable, as it can slow down the code. In this particular
case, we can avoid these calls by passing the index to the function
directly. This prevents unnecessary device-host communitation. Changes are available
in
[this](https://github.com/pytorch-tpu/diffusers/commit/0243d2ef9c2c7bc06956bb1bcc92c23038f6519d)
commit. The code now is ready to be run on TPUs.

## Running on Multiple TPU Devices

To use multiple TPU devices, use the `torch_xla.launch` function
to run the function on multiple devices and sync when necessary.
The `torch_xla.launch` function will start processes on multiple TPU
devices and sync them when needed. This can be done by passing the
`index` argument to the function that runs on a single device. For
example,

``` python
import torch_xla

def my_function(index):
  # function that runs on a single device

torch_xla.launch(my_function, args=(0,))
```

In this example, the `my_function` function will be run on 4 TPU
devices (for a v4-8 TPU slice). Each device is assigned an index from 0 to 3.
By default, `launch()` will run the function on all
TPU devices. If you want a single process, set `debug_single_process=True`:
`launch(..., debug_single_process=True)`.

[This
file](https://github.com/ssusie/diffusers/blob/main/examples/text_to_image/inference_tpu_multidevice.py)
illustrates how xmp.spawn can be used to run stable diffusion 2.1
version on multiple TPU devices. For this example, changes were made to the
[pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)
file.

## Running on Pods

Once you have the code for running on a single host device, there is no
further change needed. You can create the TPU pod, for example, by
following these
[instructions](https://cloud.google.com/tpu/docs/pytorch-pods#create-tpu-vm).
Then run your script with

``` bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="python3 your_script.py"
```

## Reference implementations

The [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes)
repo contains examples for training and serving many LLM and diffusion models.
