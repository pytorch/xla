#pragma once

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace xla {

class XlaBuilder;
class XlaOp;

class XlaComputation {
 public:
  XlaComputation() = default;

  XlaComputation(const XlaOp& root, XlaBuilder* builder);

  StatusOr<ProgramShape> GetProgramShape() const;

  const HloModuleProto& proto() const {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  struct CodeGen {
    // The full loop is split into multiple shards with adjusted start and end
    // indices to allow for parallelism.
    std::vector<std::shared_ptr<torch::jit::tensorexpr::CodeGen>>
        codegen_shards;
    absl::optional<size_t> parameter_number;
    std::unordered_map<size_t, size_t> output_to_input_aliases_;
  };

  const CodeGen& codegen(size_t idx) const {
    XLA_CHECK_LT(idx, codegen_.size());
    return codegen_[idx];
  }

 private:
  std::shared_ptr<torch::jit::tensorexpr::KernelArena> kernel_arena_;
  std::vector<CodeGen> codegen_;
  ProgramShape program_shape_;
};

xla::PrimitiveType ScalarToPrimitiveType(
    torch::jit::tensorexpr::ScalarType scalar_type);

}  // namespace xla
