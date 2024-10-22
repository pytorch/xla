# pylint: disable
import os
from typing import Any, Tuple

from jax.experimental import jax2tf
import tensorflow as tf
import torch
from torch_xla2 import export


def exported_program_to_tf_function(ep, enable_xla=True):
  weights, jax_program = export.exported_program_to_jax(ep)
  wrapped = lambda *args: jax_program(weights, (args,))
  avals = export.extract_avals(ep)
  input_signature = [
      tf.TensorSpec(shape=t.shape, dtype=t.dtype, name=f"args_{i}")
      for i, t in enumerate(avals)
  ]
  tf_f = tf.function(
      jax2tf.convert(
          wrapped,
          with_gradient=False,
          enable_xla=enable_xla,
      ),
      autograph=False,
      input_signature=input_signature,
  )
  return tf_f


def exported_program_to_tf_module(ep: torch.export.ExportedProgram,
                                  enable_xla=True) -> tf.Module:
  tfm = tf.Module()
  tfm.f = exported_program_to_tf_function(ep, enable_xla)
  return tfm


def save_exported_program_as_tf_saved_model(
    ep: torch.export.ExportedProgram,
    saved_model_dir: os.PathLike,
    serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    function_alias: str = "",
    enable_xla=True,
):
  """This function will export and save a pytorch ExportedProgram to tf.saved_model format.

  The resulting tf.saved_model can be used inference using tf.serving model
  server
  or further convert to tflite flatbuffer for on-device serving.

  Args:
    torch_model: torch.nn.Module - model to export and save
    args: Tuple[Any] - a set of args to trace the model with, i.e.
      torch_model(*args) must run
    saved_model_dir: os.PathLike - location to an empty directory to store the
      saved_model
    serving_key: str  - serving key tag, this is used by tf.serving to know
      which function to run.
    function_alias: str - passed through saved_model.save, used to tag a
      function for inference converter or other tools.
  """
  tfm = exported_program_to_tf_module(ep, enable_xla=enable_xla)
  signatures = {
      serving_key: tfm.f.get_concrete_function(*tfm.f.input_signature)
  }
  save_options = tf.saved_model.SaveOptions(function_aliases={
      function_alias: tfm.f,
  })
  tf.saved_model.save(
      tfm,
      saved_model_dir,
      signatures=signatures,
      options=save_options,
  )


def save_torch_module_as_tf_saved_model(
    torch_model: torch.nn.Module,
    args: Tuple[Any],
    saved_model_dir: os.PathLike,
    serving_key: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    function_alias: str = "",
    enable_xla=True,
):
  """This function will export and save a pytorch nn.Module to tf.saved_model format.

  The resulting tf.saved_model can be used inference using tf.serving model
  server
  or further convert to tflite flatbuffer for on-device serving.

  Args:
    torch_model: torch.nn.Module - model to export and save
    args: Tuple[Any] - a set of args to trace the model with, i.e.
      torch_model(*args) must run
    saved_model_dir: os.PathLike - location to an empty directory to store the
      saved_model
    serving_key: str  - serving key tag, this is used by tf.serving to know
      which function to run.
    function_alias: str - passed through saved_model.save, used to tag a
      function for inference converter or other tools.
  """
  ep = torch.export.export(torch_model, args)
  save_exported_program_as_tf_saved_model(ep, saved_model_dir, serving_key,
                                          function_alias, enable_xla)


def exported_program_to_tflite_flatbuffer(ep: torch.export.ExportedProgram):
  tfm = exported_program_to_tf_module(ep)
  tf_concrete_func = tfm.f.get_concrete_function(*tfm.f.input_signature)
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_concrete_func], tfm)
  tflite_model = converter.convert()
  return tflite_model


def torch_module_to_tflite_flatbuffer(torch_model: torch.nn.Module,
                                      args: Tuple[Any]):
  ep = torch.export.export(torch_model, args)
  return exported_program_to_tflite_flatbuffer(ep)
