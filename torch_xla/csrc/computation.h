#pragma once

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/xla_client/types.h"

namespace torch_xla {

class Computation {
 public:
  Computation(std::string name, xla::XlaComputation computation);

  const std::string& name() const { return name_; }

  const xla::XlaComputation& computation() const { return computation_; }

  const xla::ProgramShape& program_shape() const { return program_shape_; }

  const xla::hash_t& hash() const { return hash_; }

 private:
  std::string name_;
  xla::XlaComputation computation_;
  xla::ProgramShape program_shape_;
  xla::hash_t hash_;
};

using ComputationPtr = std::shared_ptr<Computation>;

}  // namespace torch_xla
