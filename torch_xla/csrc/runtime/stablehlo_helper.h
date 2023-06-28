#ifndef STABLEHLO_HELPER_H_
#define STABLEHLO_HELPER_H_

#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace mlir {
class ModuleOp;
}  // namespace mlir

namespace torch_xla {
namespace runtime {

std::string hloToStablehlo(const xla::HloModuleProto* proto,
                           bool emit_bytecode);

void ConvertHloToStableHlo(const xla::HloModuleProto* proto,
                           mlir::ModuleOp* mlir_module);

}  // namespace runtime
}  // namespace torch_xla

#endif
