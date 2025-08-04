# pylint: disable
import os
from typing import Any, Tuple

from jax.experimental import jax2tf
import tensorflow as tf
import torch
from torchax import export


def exported_program_to_tf_function(ep, enable_xla=True):
  """Converts a `torch.export.ExportedProgram` to a TensorFlow function.

  This function takes a PyTorch `ExportedProgram`, converts it to a JAX program,
  and then wraps it as a TensorFlow function using `jax2tf`.

  **Arguments:**

  *   `ep` (`torch.export.ExportedProgram`): The PyTorch `ExportedProgram` to convert.
  *   `enable_xla` (`bool`, optional): Whether to enable XLA compilation for the
      converted TensorFlow function. Defaults to `True`.

  **Returns:**

  A TensorFlow function that is equivalent to the input `ExportedProgram`.
  """
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
  """Converts a `torch.export.ExportedProgram` to a `tf.Module`.

  This function wraps the TensorFlow function created by
  `exported_program_to_tf_function` in a `tf.Module` for easier use and saving.

  **Arguments:**

  *   `ep` (`torch.export.ExportedProgram`): The PyTorch `ExportedProgram` to convert.
  *   `enable_xla` (`bool`, optional): Whether to enable XLA compilation. Defaults
      to `True`.

  **Returns:**

  A `tf.Module` containing the converted TensorFlow function.
  """
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
  """Exports and saves a PyTorch `ExportedProgram` to the TensorFlow SavedModel format.

  The resulting SavedModel can be used for inference with TensorFlow Serving or
  further converted to TFLite for on-device deployment.

  **Arguments:**

  *   `ep` (`torch.export.ExportedProgram`): The PyTorch `ExportedProgram` to save.
  *   `saved_model_dir` (`os.PathLike`): The path to an empty directory where the
      SavedModel will be stored.
  *   `serving_key` (`str`, optional): The serving key to use for the signature
      definition. This is used by TensorFlow Serving to identify the function
      to run. Defaults to `tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY`.
  *   `function_alias` (`str`, optional): An alias for the function, which can be
      used by other tools.
  *   `enable_xla` (`bool`, optional): Whether to enable XLA compilation. Defaults
      to `True`.
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
  """Exports and saves a `torch.nn.Module` to the TensorFlow SavedModel format.

  This function first exports the `torch.nn.Module` to an `ExportedProgram`
  and then saves it as a SavedModel.

  **Arguments:**

  *   `torch_model` (`torch.nn.Module`): The PyTorch model to export and save.
  *   `args` (`Tuple[Any]`): A tuple of arguments to trace the model with (i.e.,
      `torch_model(*args)` must be a valid call).
  *   `saved_model_dir` (`os.PathLike`): The path to an empty directory where the
      SavedModel will be stored.
  *   `serving_key` (`str`, optional): The serving key for the signature
      definition.
  *   `function_alias` (`str`, optional): An alias for the function.
  *   `enable_xla` (`bool`, optional): Whether to enable XLA compilation.
  """
  ep = torch.export.export(torch_model, args)
  save_exported_program_as_tf_saved_model(ep, saved_model_dir, serving_key,
                                          function_alias, enable_xla)


def exported_program_to_tflite_flatbuffer(ep: torch.export.ExportedProgram):
  """Converts a `torch.export.ExportedProgram` to a TFLite flatbuffer.

  **Arguments:**

  *   `ep` (`torch.export.ExportedProgram`): The PyTorch `ExportedProgram` to convert.

  **Returns:**

  A TFLite flatbuffer model.
  """
  tfm = exported_program_to_tf_module(ep)
  tf_concrete_func = tfm.f.get_concrete_function(*tfm.f.input_signature)
  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [tf_concrete_func], tfm)
  tflite_model = converter.convert()
  return tflite_model


def torch_module_to_tflite_flatbuffer(torch_model: torch.nn.Module,
                                      args: Tuple[Any]):
  """Converts a `torch.nn.Module` to a TFLite flatbuffer.

  **Arguments:**

  *   `torch_model` (`torch.nn.Module`): The PyTorch model to convert.
  *   `args` (`Tuple[Any]`): A tuple of arguments to trace the model with.

  **Returns:**

  A TFLite flatbuffer model.
  """
  ep = torch.export.export(torch_model, args)
  return exported_program_to_tflite_flatbuffer(ep)
