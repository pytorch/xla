import itertools
import re
import sys
import os
from typing import List, Tuple, Any
import copy
import logging

import torch
from torch_xla import stablehlo

try:
  import tensorflow as tf
  from tensorflow.compiler.tf2xla.python import xla as tfxla
except ImportError:
  logging.error('This module is need tensorflow with xla support.\n'
                'Please install tensorflow with `pip install tf-nightly`.\n')
  raise


def _get_shape_with_dynamic(signature: stablehlo.VariableSignature):
  shape = copy.copy(signature.shape)
  for i in signature.dynamic_dims:
    shape[i] = None
  return shape


def _wrap_as_tf_func(func, bundle):

  def inner(*args):
    output_sig = func.meta.output_signature[0]
    Touts = [sig.dtype for sig in func.meta.output_signature]
    Souts = [_get_shape_with_dynamic(sig) for sig in func.meta.output_signature]
    call_args = stablehlo._extract_call_parameters(args, func.meta, bundle)
    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=Touts,  # dtype information
        Sout=Souts,  # Shape information
        function_list=[],
        module=func.bytecode,
    )

  return inner


def make_tf_function(stablehlo_program: stablehlo.StableHLOGraphModule,
                     bundle=None):
  if bundle is None:
    return _wrap_as_tf_func(stablehlo_program._bundle.stablehlo_funcs[0],
                            stablehlo_program._bundle)
  return _wrap_as_tf_func(stablehlo_program._bundle.stablehlo_funcs[0], bundle)


def _make_input_signatures(
    meta: stablehlo.StableHLOFunctionMeta) -> List[tf.TensorSpec]:
  input_pos_to_spec = {
      loc.position: spec
      for loc, spec in itertools.chain(
          zip(meta.input_locations, meta.input_signature), meta.unused_inputs)
      if loc.type_ == stablehlo.VariableType.INPUT_ARG
  }
  primitive_type_to_tf_type = {'int': 'int32', 'float': 'float32'}
  for i in range(len(input_pos_to_spec)):
    spec = input_pos_to_spec[i]
    shape = _get_shape_with_dynamic(spec)
    yield tf.TensorSpec(
        shape=shape,
        dtype=getattr(
            tf, primitive_type_to_tf_type[spec.dtype]
            if spec.dtype in primitive_type_to_tf_type else spec.dtype),
        name=f'args_{i}')


def _mangle_tf_root_scope_name(name):
  # TF has more restricted constrain on the variable names at root scope.
  # Root scope name constrain: [A-Za-z0-9.][A-Za-z0-9_.\\-/]*
  # Non-root scope name constrain: [A-Za-z0-9_.\\-/]*
  # https://github.com/tensorflow/tensorflow/blob/51b601fa6bb7e801c0b6ae73c25580e40a8b5745/tensorflow/python/framework/ops.py#L3301-L3302
  # The state_dict key doesn't have such constrain,
  # the name need to be mangled when a root-scoped TF variable is created.
  #
  # FX Graph Node may contain characters other than [A-Za-z0-9_.\\-/], replace
  # offending characters with '_'.
  if name[0] in "._\\-/":
    name = 'k' + name
  name = re.sub(r"[^^\w\-/\\]+", "_", name)
  return name


def save_stablehlo_graph_as_tf(
    stablehlo_program: stablehlo.StableHLOGraphModule,
    path: os.PathLike,
    serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    function_alias: str = '') -> None:
  """This function will export and save a StableHLOGraphModule to tf.saved_model format.

  The resulting tf.saved_model can be used inference using tf.serving model server
  or further convert to tflite flatbuffer for on-device serving.

  StableHLOGraphModule is produced with the torch_xla.stablehlo package.

  Args:
    stablehlo_program - model to export and save
    path: os.PathLike - location to an empty directory to store the saved_model
    serving_key: str  - serving key tag, this is used by tf.serving to know which function to run.
    function_alias: str - passed through saved_model.save, used to tag a function for 
       inference converter or other tools.
  """

  bundle = copy.deepcopy(stablehlo_program._bundle)
  tfm = tf.Module()
  bundle.state_dict = {
      k: tf.Variable(v, trainable=False, name=_mangle_tf_root_scope_name(k))
      for k, v in bundle.state_dict.items()
  }
  bundle.additional_constants = [
      tf.Variable(v, trainable=False) for v in bundle.additional_constants
  ]
  input_signatures = list(
      _make_input_signatures(bundle.stablehlo_funcs[0].meta))
  tfm.f = tf.function(
      make_tf_function(stablehlo_program, bundle),
      input_signature=input_signatures)
  tfm._variables = (
      list(bundle.state_dict.values()) + bundle.additional_constants)
  signatures = {serving_key: tfm.f.get_concrete_function(*input_signatures)}
  save_options = tf.saved_model.SaveOptions(function_aliases={
      function_alias: tfm.f,
  })
  tf.saved_model.save(
      tfm,
      path,
      signatures=signatures,
      options=save_options,
  )


def save_torch_module_as_tf_saved_model(
    torch_model: torch.nn.Module,
    args: Tuple[Any],
    saved_model_dir: os.PathLike,
    dynamic_shapes=None,
    serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    function_alias: str = '',
):
  """This function will export and save a pytorch nn.Module to tf.saved_model format.

  The resulting tf.saved_model can be used inference using tf.serving model server
  or further convert to tflite flatbuffer for on-device serving.

  Args:
    torch_model: torch.nn.Module - model to export and save
    args: Tuple[Any] - a set of args to trace the model with, i.e. torch_model(*args) must run
    saved_model_dir: os.PathLike - location to an empty directory to store the saved_model
    serving_key: str  - serving key tag, this is used by tf.serving to know which function to run.
    function_alias: str - passed through saved_model.save, used to tag a function for 
       inference converter or other tools.
  """
  exported = torch.export.export(
      torch_model, args, dynamic_shapes=dynamic_shapes)
  options = stablehlo.StableHLOExportOptions(override_tracing_arguments=args)
  stablehlo_model = stablehlo.exported_program_to_stablehlo(exported, options)
  save_stablehlo_graph_as_tf(stablehlo_model, saved_model_dir, serving_key,
                             function_alias)


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'input_dir',
      help='Directory with the output of torch_xla.save_as_stablehlo')
  parser.add_argument(
      'output_dir',
      help='Empty Directory which we will create tf.saved_model in.')
  parser.add_argument(
      '-a',
      '--function_alias',
      help='Apply function alias to the serving function. This '
      'is usually used by inference_converter or other tools.',
      default='')
  parser.add_argument(
      '-k',
      '--serving_key',
      help='Apply serving key the serving function. This '
      'is usually used by tf.serving model serving or other tools.'
      'Default to "serving_default".',
      default=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
  )
  args = parser.parse_args()

  shlo_program = stablehlo.StableHLOGraphModule.load(args.input_dir)
  print('Loading {} into StableHLOGraphModule...'.format(args.input_dir))
  shlo_program = stablehlo.StableHLOGraphModule.load(args.input_dir)
  print('Saving as saved model to {} ...'.format(args.output_dir))
  save_stablehlo_graph_as_tf(shlo_program, args.output_dir, args.serving_key,
                             args.function_alias)
  print('Done!')
  return 0


if __name__ == '__main__':
  sys.exit(main())
