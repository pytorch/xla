#pragma once

#include <string>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

struct XlaComputationInOut {
  std::vector<xla::XlaOp> inputs;
  std::vector<xla::XlaOp> outputs;
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

  XlaTranslator(const std::shared_ptr<Graph>& graph,
                const xla::PrecisionConfig::Precision conv_precision);

  // Builds and compiles the XLA computation for graph_.
  xla::XlaComputation BuildComputation(
      const std::string& name,
      const std::vector<ParameterShape>& parameter_shapes,
      const BuildOptions& options = BuildOptions()) const;

  // Builds the XLA computation for graph_ without compiling it and returns the
  // XLA operations for inputs and outputs.
  XlaComputationInOut BuildComputationProgram(
      const std::vector<ParameterShape>& parameter_shapes,
      xla::XlaBuilder* b) const;

 private:
  std::shared_ptr<Graph> graph_;
  xla::PrecisionConfig::Precision conv_precision_;
};

xla::ComputationClient* XlaGetClient();

}  // namespace jit
}  // namespace torch
