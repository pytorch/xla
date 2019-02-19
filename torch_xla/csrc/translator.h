#pragma once

#include <string>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

struct XlaComputationInOut {
  using SizeOpValues = std::unordered_map<size_t, std::vector<xla::int64>>;
  std::vector<xla::XlaOp> inputs;
  std::vector<xla::XlaOp> outputs;
  // Stores the values for return components which are the result of aten::size
  // evaluation. Keys are the component indices inside the return tuple.
  SizeOpValues ret_size_op_values;
};

// The result of translation to XLA: the computation and the map of constant
// aten::size values in the return tuple. See XlaComputationInOut for the
// description of ret_size_op_values.
struct XlaTranslationResult {
  xla::XlaComputation computation;
  XlaComputationInOut::SizeOpValues ret_size_op_values;
};

class XlaTranslator {
 public:
  enum class ParameterKind {
    kGraphInput,
    kZeroInput,
  };

  struct ParameterShape {
    // A shape created with zero_input == true, when passed to the
    // BuildComputation*() APIs, will generate an artificial zero input (of
    // proper shape) value for the XLA computation.
    ParameterShape(xla::Shape shape, ParameterKind kind)
        : shape(std::move(shape)), kind(kind) {}

    xla::Shape shape;
    ParameterKind kind;
  };

  struct BuildOptions {
    BuildOptions() {}

    // Optional transfor function which is called to apply transformation to the
    // computation outputs before they get merged into the output tuple.
    std::function<xla::XlaOp(const xla::XlaOp&, size_t)> output_transform;
  };

  XlaTranslator(const std::shared_ptr<torch::jit::Graph>& graph);

  // Builds and compiles the XLA computation for graph_. For the backward
  // computation, param_size_op_values stores the constant values for aten::size
  // from the forward computation.
  XlaTranslationResult BuildComputation(
      const std::string& name,
      const std::vector<ParameterShape>& parameter_shapes,
      const XlaComputationInOut::SizeOpValues& param_size_op_values,
      const BuildOptions& options = BuildOptions()) const;

  // Builds the XLA computation for graph_ without compiling it and returns the
  // XLA operations for inputs and outputs.
  XlaComputationInOut BuildComputationProgram(
      const std::vector<ParameterShape>& parameter_shapes,
      const XlaComputationInOut::SizeOpValues& param_size_op_values,
      xla::XlaBuilder* b) const;

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
};

}  // namespace torch_xla
