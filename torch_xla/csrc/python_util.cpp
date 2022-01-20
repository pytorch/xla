#include "torch_xla/csrc/python_util.h"

#include <Python.h>
#include <frameobject.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch_xla {

absl::optional<torch::lazy::SourceLocation> GetPythonFrameTop() {
  if (!Py_IsInitialized()) {
    return absl::nullopt;
  }
  pybind11::gil_scoped_acquire gil;
  PyFrameObject* frame = PyEval_GetFrame();
  if (frame == nullptr) {
    return absl::nullopt;
  }
  torch::lazy::SourceLocation loc;
  loc.line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
  loc.file = THPUtils_unpackString(frame->f_code->co_filename);
  loc.function = THPUtils_unpackString(frame->f_code->co_name);
  return loc;
}

std::vector<torch::lazy::SourceLocation> GetPythonFrames() {
  std::vector<torch::lazy::SourceLocation> frames;
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    PyFrameObject* frame = PyEval_GetFrame();
    while (frame != nullptr) {
      torch::lazy::SourceLocation loc;
      loc.line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
      loc.file = THPUtils_unpackString(frame->f_code->co_filename);
      loc.function = THPUtils_unpackString(frame->f_code->co_name);
      frames.push_back(std::move(loc));
      frame = frame->f_back;
    }
  }
  return frames;
}

}  // namespace torch_xla
