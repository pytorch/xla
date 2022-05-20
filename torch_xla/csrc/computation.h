#pragma once

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/types.h"
#include "torch/csrc/lazy/backend/lowering_context.h"
#include "torch/csrc/lazy/core/hash.h"

namespace torch_xla {

class Computation : public torch::lazy::Computation {
 public:
  Computation(std::string name, xla::XlaComputation computation);

  const std::string& name() const { return name_; }

  const xla::XlaComputation& computation() const { return computation_; }

  // We don't want to make a copy when passing computation_ to the runtime.
  // Class member will be accessed as const& and `xla::XlaComputation`
  // explictly delete its const& copy constructor so we have to const cast here.
  xla::XlaComputation move_computation() const {
    return std::move(const_cast<Computation*>(this)->computation_);
  }

  const xla::ProgramShape& program_shape() const { return program_shape_; }

  const torch::lazy::hash_t& hash() const { return hash_; }

  int parameters_size() const override {
    return program_shape_.parameters_size();
  }

  const std::vector<torch::lazy::Shape>& parameter_shapes() const override {
    // TODO: convert the program_shape_.parameters() back to torch::lazy::Shape
    return parameter_shapes_;
  }

  const std::vector<std::string>& parameter_names() const override {
    return program_shape_.parameter_names();
  }

  const torch::lazy::Shape& result_shape() const override {
    // TODO: convert the program_shape_.result() back to torch::lazy::Shape
    return res_shape_;
  }

  const std::string to_string() const override {
    xla::HloModuleConfig hlo_config(program_shape_);
    std::unique_ptr<xla::HloModule> module = ConsumeValue(
        xla::HloModule::CreateFromProto(computation_.proto(), hlo_config));
    return module->ToString();
  }

 private:
  std::string name_;
  xla::ProgramShape program_shape_;
  xla::XlaComputation computation_;
  torch::lazy::hash_t hash_;
  torch::lazy::Shape res_shape_;
  std::vector<torch::lazy::Shape> parameter_shapes_;
};

using ComputationPtr = std::shared_ptr<Computation>;

}  // namespace torch_xla
