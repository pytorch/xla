# LD_LIBRARY_PATH=~/miniconda3/envs/torch310/lib/:/usr/lib/x86_64-linux-gnu/ PJRT_DEVICE=CPU python xla/generate_stablehlo.py
import copy
import shutil
import os
import re
import numpy as np
import torch
from torch import nn
import torch_xla
from torch_xla.core import xla_model as xm
import jax
import tensorflow as tf
import torchvision
import torch._dynamo as torchdynamo

from tensorflow.compiler.tf2xla.python import xla as tfxla  # type: ignore[import]

from typing import Tuple, Type, Callable

from jax.experimental.jax2tf import jax2tf
from jax.experimental.jax2tf import jax_export
from jax._src.lib import xla_client
from jax._src.interpreters import mlir
from jax._src.sharding_impls import GSPMDSharding
from jax import tree_util

import sys
# MLIR python bindings
# sys.path.append('/home/hanq/git/llvm/llvm-project/build/tools/mlir/python_packages/mlir_core')

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import stablehlo

def convert_mlir(bytecode, sample_inputs, sample_outputs):
    output_tf = tf.convert_to_tensor(sample_outputs)
    Tout = (output_tf.dtype, )
    Sout = (tuple(output_tf.shape), )
    
    def return_func(*args):
        return tfxla.call_module(
            args,
            version=5,
            Tout=Tout,  # dtype information
            Sout=Sout,  # Shape information
            function_list=[],
            platforms=('CPU', ),
            module=bytecode,
        )
    return return_func

def fix_mlir_tuple(mlir_module):
    print(mlir_module.body.operations[0].type)
    type_ = mlir_module.body.operations[0].type
    for inst in mlir_module.body.operations[0].body.blocks[0].operations:
        print(inst)

    if isinstance(type_.results[0], ir.TupleType):
        with mlir_module.context:
            mlir_module.body.operations[0].attributes["function_type"] = ir.FunctionType.get(
                type_.inputs, [type_.results[0].get_type(0)]
            )
    
    mlir_module.dump()
    return mlir_module

tuple_regex = re.compile(r'tuple<(.*)>')
xla_shape = re.compile(r'{xla_shape = .*}')
def fix_hlo_text(hlotext):
    ## HACK
    res = []
    for line in hlotext.split('\n'):
        if m := tuple_regex.findall(line):
            line = tuple_regex.sub(m[0], line)
        line = line.replace('stablehlo.tuple', 'stablehlo.optimization_barrier')
        line = xla_shape.sub('', line)
        res.append(line)
    return '\n'.join(res)


class SHLOModel:

    def __init__(self, model_bytecode, pos_to_orig_pos, pos_to_param, sample_output):
        self.model_bytecode = model_bytecode
        self.pos_to_orig_pos = pos_to_orig_pos
        self.pos_to_param = pos_to_param
        self._total_number = len(pos_to_orig_pos) + len(pos_to_param)
        output_tf = tf.convert_to_tensor(sample_output)
        self.tout = (output_tf.dtype, )
        self.sout = (tuple(output_tf.shape), )

    def __call__(self, *args):

        call_args = []
        for i in range(self._total_number):
            if i in self.pos_to_orig_pos:
                call_args.append(args[self.pos_to_orig_pos[i]])
            else:
                call_args.append(self.pos_to_param[i])

        return tfxla.call_module(
            tuple(call_args),
            version=5,
            Tout=self.tout,  # dtype information
            Sout=self.sout,  # Shape information
            function_list=[],
            platforms=('CPU', ),
            module=self.model_bytecode,
        )
        
        



def export_torch_model(model: torch.nn.Module, sample_inputs: Tuple):

    inputs_torch = tuple(map(torch.tensor, sample_inputs))
    sample_outputs = model(*inputs_torch)

    device = xm.xla_device()
    model.to(device=device)
    sample_inputs_lazy = tuple(map(lambda x: x.to(device=device), inputs_torch))
    input_ids = {torch_xla._XLAC._xla_get_tensor_id(tensor): i for i, tensor in enumerate(sample_inputs_lazy)}
    output = model(*sample_inputs_lazy)
    hlo_str_readable = xm.get_stablehlo([output])
    hlo_str_readable = fix_hlo_text(hlo_str_readable)

    print("======= StableHLO =========")
    print(hlo_str_readable)
    print(" ========== ")


    (
      graph_input_tensor_ids,
      graph_input_xla_values,
    ) = torch_xla._XLAC._get_tensors_xla_device_data_node([output])


    print('===== ID mapping === ')
    print('param ids: ', graph_input_tensor_ids)
    print('input_ids: ', input_ids)
    print('======== ')

    pos_to_orig_pos = {}
    pos_to_param = {}
    for hlo_input_pos, tensor_id in enumerate(graph_input_tensor_ids):
        if tensor_id in input_ids:  # this is input
            pos_to_orig_pos[hlo_input_pos] = input_ids[tensor_id]
        else:
            pos_to_param[hlo_input_pos] = graph_input_xla_values[hlo_input_pos].detach().cpu().numpy()


    # MLIR module
    mlir_module = ir.Module.parse(hlo_str_readable, context=mlir.make_ir_context())

    # module -> bytecode
    xla_call_module_version = 5
    mlir_str = mlir.module_to_bytecode(mlir_module)
    # target_version = stablehlo.get_earliest_forward_compatible_version()
    target_version = '0.9.0'
    mlir_module_serialized = xla_client._xla.mlir.serialize_portable_artifact(
        mlir_str, target_version)

    # serialized mlir to Exported object

    sample_outputs = sample_outputs.detach().numpy()
    tf_f = SHLOModel(mlir_module_serialized, pos_to_orig_pos, pos_to_param, sample_outputs)
    return tf_f


class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

class Model2(nn.Module):

    def forward(self, x, y):
        return torch.ops.aten.cos(x) + y

@torch.no_grad()
def capture(f, args):
    with torchdynamo.config.patch({
        "dynamic_shapes": True,
        "capture_scalar_outputs": True,
        "guard_nn_modules": True,
        "specialize_int": True,
        "allow_rnn": True,
        "verbose": True,
    }):
        graphmodule, _ = torchdynamo.export(
            f,
            *copy.deepcopy(args),
            aten_graph=True,
        )
        return graphmodule


def main3():
    model = BasicBlock(3, 3).eval()
    inputs = (np.random.random((10, 3, 100, 100)).astype(np.float32), )
    tf_f = export_torch_model(model, inputs)
    print(tf_f(*inputs))

def main2():
    model = Model2()
    inputs = (np.random.random((10, 3, 100, 100)).astype(np.float32), 
              np.random.random((10, 3, 100, 100)).astype(np.float32), )
        
    tf_f, parameters = export_torch_model(model, inputs)
    # print(tf_f(*inputs))

def main():
    # Set random seed
    torch.manual_seed(1234)
    np.random.seed(1234)

    model = torchvision.models.resnet18().eval()
    inputs = (np.random.random((10, 3, 224, 224)).astype(np.float32), )
    torch_out = model(torch.tensor(inputs[0])).detach().numpy()

    graphmodule = capture(model, (torch.tensor(inputs[0]), ))

    tf_f = export_torch_model(graphmodule, inputs)
    print("======== inference using Tf ======")
    converted_out = tf_f(*inputs)
    print(converted_out)
    print("==============")

    path = '/tmp/tempsavedmodel'
    print("Saving to savedmodel: ", path)
    if not os.path.exists(path):
        os.makedirs(path)

    my_model = tf.Module()
    my_model.f = tf.function(tf_f, autograph=False, input_signature=[tf.TensorSpec([10, 3, 224, 224], tf.float32)])
    vars = tf.nest.map_structure(tf.Variable, tf_f.pos_to_param)
    my_model._variables = tf.nest.flatten(vars)
    tf.saved_model.save(my_model, path)

    # Load saved model and inference
    loadedm = tf.saved_model.load(path)
    saved_model_out = loadedm.f(*inputs)
    saved_model_out = saved_model_out[0].numpy()

    # Compare torch output and tf output
    l2_norm = np.linalg.norm(torch_out - saved_model_out, ord=2)
    print("output shape {}".format(torch_out.shape))
    print("L2 distance between torch output and tf output {}".format(l2_norm))
    print("Max abs error: {}".format(np.max(torch_out - saved_model_out)))


# tuple< ? > -> ?
# stablehlo.tuple -> stablehlo.optimization_barrier


if __name__ == '__main__':
    main()