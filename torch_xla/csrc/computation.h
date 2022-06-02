#pragma once

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/types.h"
#include "torch/csrc/lazy/backend/lowering_context.h"
#include "torch/csrc/lazy/core/hash.h"

namespace torch_xla {

// Computation inherits torch::lazy::Computation is being used for three
// purposes.
// 1. To represent a xla computation build by xla_op_builder, in which case we
// would need the name and hash. Computation would be a wrapper around a
// xla::ComputationClient::Computation.
// xla::ComputationClient::Computation::devices_ would be empty.
// 2. To represent a computation built by syncTensor and needs to be compiled.
// In this case hash_ and name_ are not required. Computation would be a wrapper
// around a xla::ComputationClient::Computation.
// xla::ComputationClient::Computation::devices_ would be empty.
// 3. To represent a computation that is already compiled. In this case name_
// and hash_ are not required in this case. Computation will be a wrapper around
// xla::XrtComputationClient::XrtComputation or
// xla::PjRtComputationClient::PjRtComputation which contains a handle to
// represent the compiled program.
// It is not ideal to use same class for 3 different purposes but this is the
// path took by upstream ltc.
class Computation : public torch::lazy::Computation {
 public:
  Computation(std::string name, xla::XlaComputation computation);

  Computation(std::shared_ptr<xla::ComputationClient::Computation>
                  xla_client_computation);

  const std::string& name() const { return name_; }

  std::shared_ptr<xla::ComputationClient::Computation> client_computation() const {
    return xla_client_computation_;
  }

  const xla::XlaComputation& computation() const {
    return xla_client_computation_->computation();
  }

  // After the move, computation will become invalid and should not be
  // accessed.
  xla::XlaComputation move_computation() const {
    return const_cast<Computation*>(this)
        ->xla_client_computation_->move_computation();
  }

  const xla::ProgramShape& program_shape() const {
    return xla_client_computation_->program_shape();
  }

  const torch::lazy::hash_t& hash() const { return hash_; }

  int parameters_size() const override {
    return program_shape().parameters_size();
  }

  const std::vector<torch::lazy::Shape>& parameter_shapes() const override {
    // TODO: convert the program_shape().parameters()
    // back to torch::lazy::Shape
    return parameter_shapes_;
  }

  const std::vector<std::string>& parameter_names() const override {
    return program_shape().parameter_names();
  }

  const torch::lazy::Shape& result_shape() const override {
    // TODO: convert the program_shape() back to
    // torch::lazy::Shape
    return res_shape_;
  }

  const std::string to_string() const override {
    xla::HloModuleConfig hlo_config(program_shape());
    std::unique_ptr<xla::HloModule> module = ConsumeValue(
        xla::HloModule::CreateFromProto(computation().proto(), hlo_config));
    return module->ToString();
  }

 private:
  std::string name_;
  std::shared_ptr<xla::ComputationClient::Computation> xla_client_computation_;
  torch::lazy::hash_t hash_;
  torch::lazy::Shape res_shape_;
  std::vector<torch::lazy::Shape> parameter_shapes_;
};

using ComputationPtr = std::shared_ptr<Computation>;

std::vector<torch::lazy::ComputationPtr> WrapClientComputation(
    std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
        computations);

std::shared_ptr<xla::ComputationClient::Computation> UnwrapClientComputation(
    torch::lazy::ComputationPtr computations);

}  // namespace torch_xla
