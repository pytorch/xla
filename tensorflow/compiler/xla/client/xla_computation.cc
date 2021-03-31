#include "tensorflow/compiler/xla/client/xla_computation.h"

#include "lazy_tensors/computation_client/nnc_computation_client.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"

using namespace torch::jit::tensorexpr;

namespace xla {
namespace {

Stmt* OptimizeComputation(LoopNest* l, absl::Span<Tensor*> temporaries) {
  for (const auto temporary : temporaries) {
    if (!l->hasLoopBodyFor(temporary)) {
      continue;
    }
    auto temporary_statement = l->getLoopBodyFor(temporary);
    l->computeInline(temporary_statement);
  }
  l->prepareForCodegen();
  return IRSimplifier::simplify(l->root_stmt());
}

// Get the optimized, ready for code generation statement for the given tensor
// and temporaries.
Stmt* GetComputation(Tensor* tensor, absl::Span<Tensor*> temporaries) {
  LoopNest l({tensor});
  return OptimizeComputation(&l, temporaries);
}

// Returns the immediate loop bounds, if available.
absl::optional<std::pair<int, int>> ImmediateLoopBounds(For* f) {
  const auto start_imm = dynamic_cast<const IntImm*>(f->start());
  const auto stop_imm = dynamic_cast<const IntImm*>(f->stop());
  if (!start_imm || !stop_imm) {
    return absl::nullopt;
  }
  return std::make_pair(start_imm->value(), stop_imm->value());
}

// Generate multiple computations for the given tensor and temporaries, each
// covering a contiguous range of the original, full range. This is required to
// achieve inter-operator parallelism.
std::vector<Stmt*> GetComputationSlices(Tensor* tensor,
                                        absl::Span<Tensor*> temporaries) {
  // Retrieve the number of cores from the environment for now.
  // TODO(asuhan): Use a sensible default instead once we have more reliable
  // measurements.
  static const int kNumCores =
      lazy_tensors::sys_util::GetEnvInt("NNC_NUM_CORES", 1);
  if (kNumCores == 1) {
    // For single core computation, bypass all the splitting work.
    return {GetComputation(tensor, temporaries)};
  }
  LoopNest l({tensor});
  OptimizeComputation(&l, temporaries);
  const auto for_loops = l.getLoopStmtsFor(tensor);
  // Only parallelize single, fully fused loops for now.
  if (for_loops.size() != 1) {
    return {GetComputation(tensor, temporaries)};
  }
  const auto f = for_loops.front();
  auto bounds_opt = ImmediateLoopBounds(f);
  // Bounds should be always known with static shapes.
  if (!bounds_opt) {
    return {GetComputation(tensor, temporaries)};
  }
  int start_val = bounds_opt->first;
  int stop_val = bounds_opt->second;
  // Compute the chunk size for the number of cores, rounding up.
  int chunk_size =
      std::max((stop_val - start_val + kNumCores - 1) / kNumCores, 1);
  For* leftover = f;
  std::vector<Stmt*> slices;
  // Slice the head of the loop until we're left with an empty tail.
  while (true) {
    For* head{nullptr};
    For* tail{nullptr};
    l.sliceHead(leftover, chunk_size, &head, &tail);
    const auto simplified_head = IRSimplifier::simplify(head);
    slices.push_back(simplified_head);
    if (!tail) {
      break;
    }
    const auto simplified_tail = IRSimplifier::simplify(tail);
    tail = dynamic_cast<For*>(simplified_tail);
    if (!tail) {
      slices.push_back(simplified_tail);
      break;
    }
    LTC_CHECK(tail);
    // TODO(asuhan): In certain corner cases the last tail doesn't constant fold
    // properly its start and end indices, investigate / fix why that is.
    if (!ImmediateLoopBounds(tail)) {
      return {GetComputation(tensor, temporaries)};
    }
    leftover = tail;
    Block::make({leftover});
  }
  return slices;
}

}  // namespace

XlaComputation::XlaComputation(const XlaOp& root, XlaBuilder* builder)
    : kernel_arena_(builder->kernel_arena()) {
  for (const auto& output : root.outputs()) {
    const auto tensor = output.expr;
    if (!tensor) {
      LTC_CHECK(output.arg && output.arg_idx);
      codegen_.emplace_back(CodeGen{{}, *output.arg_idx});
      continue;
    }
    std::vector<Tensor*> temporaries;
    for (const XlaOp& t : builder->GetOperators()) {
      if (t.outputs().size() > 1) {
        continue;
      }
      const auto& output = t.outputs().front();
      if (output.expr == tensor) {
        continue;
      }
      temporaries.push_back(output.expr);
    }
    std::vector<torch::jit::tensorexpr::CodeGen::BufferArg> formal_parameters;
    for (const auto& parameter : builder->GetParameters()) {
      const auto& outputs = parameter.outputs();
      LTC_CHECK_EQ(outputs.size(), size_t(1));
      const auto formal_parameter = outputs.front().arg;
      LTC_CHECK(formal_parameter);
      formal_parameters.emplace_back(*formal_parameter);
    }
    formal_parameters.emplace_back(Placeholder(BufHandle(tensor->buf())));
    const auto stmt_slices =
        GetComputationSlices(tensor, absl::MakeSpan(temporaries));
    std::vector<std::shared_ptr<torch::jit::tensorexpr::CodeGen>>
        codegen_shards;
    const auto device_type =
        lazy_tensors::NNCComputationClient::HardwareDeviceType();
    try {
      switch (device_type) {
        case at::kCPU: {
          for (const auto stmt : stmt_slices) {
            codegen_shards.push_back(
                CreateCodeGen("llvm_codegen", stmt, formal_parameters));
          }
          break;
        }
        case at::kCUDA: {
          LTC_CHECK_EQ(stmt_slices.size(), size_t(1));
          codegen_shards.push_back(CreateCodeGen("cuda_codegen", stmt_slices[0],
                                                 formal_parameters,
                                                 {at::kCUDA, 0}));
          break;
        }
        default: { TF_LOG(FATAL) << "Device not supported: " << device_type; }
      }
    } catch (const std::runtime_error& error) {
      LTC_CHECK_EQ(device_type, at::kCPU);
      LOG(ERROR) << error.what();
      for (const auto stmt : stmt_slices) {
        codegen_shards.push_back(
            CreateCodeGen("simple_ir_eval", stmt, formal_parameters));
      }
    }
    codegen_.push_back(XlaComputation::CodeGen{
        codegen_shards, absl::nullopt, builder->GetOutputToInputAliases()});
  }
  const auto& shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(root);
  program_shape_ = ProgramShape(shape, builder->GetParameters().size());
}

StatusOr<ProgramShape> XlaComputation::GetProgramShape() const {
  return program_shape_;
}

PrimitiveType ScalarToPrimitiveType(ScalarType scalar_type) {
  switch (scalar_type) {
    case ScalarType::Char: {
      return PrimitiveType::S8;
    }
    case ScalarType::Short: {
      return PrimitiveType::S16;
    }
    case ScalarType::Byte: {
      return PrimitiveType::U8;
    }
    case ScalarType::Int: {
      return PrimitiveType::S32;
    }
    case ScalarType::Long: {
      return PrimitiveType::S64;
    }
    case ScalarType::Float: {
      return PrimitiveType::F32;
    }
    case ScalarType::Double: {
      return PrimitiveType::F64;
    }
    case ScalarType::Bool: {
      return PrimitiveType::PRED;
    }
    default: { TF_LOG(FATAL) << "Not implemented yet."; }
  }
}

}  // namespace xla
