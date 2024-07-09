Torch Export to StableHLO (Prototype feature)
--------------------------

This document describes how to use torch export + torch xla to export to 
[StableHLO](https://github.com/openxla/stablehlo) format.

**NOTE:** Currently this is classified as prototype feature. It's API specifics 
will change in the next (2.2) release.

## How to use:

```python
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo
import torch_xla.core.xla_model as xm
import torchvision
import torch

xla_device = xm.xla_device()

resnet18 = torchvision.models.resnet18()
# Sample input is a tuple
sample_input = (torch.randn(4, 3, 224, 224), )
output = resnet18(*sample_input)
exported = export(resnet18, sample_input)
stablehlo_program = exported_program_to_stablehlo(exported)

# Now stablehlo_program is a callable backed by stablehlo IR.

# we can see it's stablehlo code with
#   here 'forward' is the name of function. Currently we only support
#   one entry point per program, but in the future we will support
#   multiple entry points in a program.
print(stablehlo_program.get_stablehlo_text('forward'))

# we can also print out the bytecode
print(stablehlo_program.get_stablehlo_bytecode('forward'))

# we can also run the module, to run the stablehlo module, we need to move 
# our tensors to XLA device.
sample_input_xla = tuple(s.to(xla_device) for s in sample_input)

output2 = stablehlo_program(*sample_input_xla) 
print(torch.allclose(output, output2.cpu(), atol=1e-5)) 
```

# Saving StableHLO bytecodes to disk

One can now save stablehlo to disk with 
```python
stablehlo_program.save('/tmp/stablehlo_dir')
```
The path should be path to an empty directory. If it doesn't exist, it will be created.
This directory can be loaded again as another stablehlo_program:

```python
from torch_xla.stablehlo import StableHLOGraphModule
stablehlo_program2 = StableHLOGraphModule.load('/tmp/stablehlo_dir')
output3 = stablehlo_program2(*sample_input_xla) 
```

# Convert saved StableHLO for serving

StableHLO is an open format and it is supported for serving in  [tensorflow.serving](https://github.com/tensorflow/serving) model server. However, before giving it to tf.serving, we need to first
wrap the generated StableHLO bytecode into a `tf.saved_model` format.

For that, first ensure that you have the latest `tensorflow` install in the current python env,
if not, install with

```bash
pip install tf-nightly
```

Now, you can run a converter (provided in the torch/xla installation)
```
stablehlo-to-saved-model /tmp/stablehlo_dir /tmp/resnet_tf/1
```

After that, you can run your model server on the newly generated `tf.saved_model` with
tf serving binary.


```
docker pull tensorflow/serving
docker run -p 8500:8500 \
--mount type=bind,source=/tmp/resnet_tf,target=/models/resnet_tf \
-e MODEL_NAME=resnet_tf -t tensorflow/serving &
```

You can also use the `tf.serving` binary directly without docker. 
For more details, please follow the [tf serving guide](https://www.tensorflow.org/tfx/serving/serving_basic).

# Common wrappers

### I want to save directly tf.saved_model format without needing to run an separate command.

You can accomplish this by using this helper function:
```python
from torch_xla.tf_saved_model_integration import save_torch_module_as_tf_saved_model

save_torch_module_as_tf_saved_model(
    resnet18,  # original pytorch torch.nn.Module 
    sample_inputs, # sample inputs used to trace
    '/tmp/resnet_tf'   # directory for tf.saved_model
)
```

### Other common wrappers

```python
def save_as_stablehlo(exported_model: 'ExportedProgram',
                      stablehlo_dir: os.PathLike,
                      options: Optional[StableHLOExportOptions] = None):
```

`save_as_stablehlo` (also aliased as `torch_xla.save_as_stablehlo`) 
takes ExportedProgram and saves StableHLO on disk. i.e.
   same as exported_program_to_stablehlo(...).save(...)

```python
def save_torch_model_as_stablehlo(
    torchmodel: torch.nn.Module,
    args: Tuple[Any],
    path: os.PathLike,
    options: Optional[StableHLOExportOptions] = None) -> None:
  """Convert a torch model to a callable backed by StableHLO.

```
takes `torch.nn.Module` and saves StableHLO on disk. i.e.
   same as torch.export.export followed by save_as_stablehlo 


# Files produced by `save_as_stablehlo`.

Inside of `/tmp/stablehlo_dir` in the example above, you will find 3 directories: `data`, `constants`, `functions`. Both data and constants will consist of tensors used by the program
saved as `numpy.ndarray` using `numpy.save`.

The functions directory will contain StableHLO bytecode, here named `forward.bytecode`, human readable StableHLO code (MLIR form) `forward.mlir`, and a JSON file specifying which weights 
and original user's input become the which positional arguments of this StableHLO function; as well
as the dtypes and shapes of every argument.


Example:
```
$ find /tmp/stablehlo_dir
./functions
./functions/forward.mlir
./functions/forward.bytecode
./functions/forward.meta
./constants
./constants/3
./constants/1
./constants/0
./constants/2
./data
./data/L__fn___layers_15_feed_forward_w2.weight
./data/L__fn___layers_13_feed_forward_w1.weight
./data/L__fn___layers_3_attention_wo.weight
./data/L__fn___layers_12_ffn_norm_weight
./data/L__fn___layers_25_attention_wo.weight
...
```

The JSON file is serialized form of the `torch_xla.stablehlo.StableHLOFunc` class.

This format is currently also in prototype stage and there are no backward compatibility guarantees.
The future plan is to standardize a format that the major frameworks (PyTorch, JAX, TensorFlow) can agree.

