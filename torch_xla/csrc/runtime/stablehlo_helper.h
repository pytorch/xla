#ifndef STABLEHLO_HELPER_H_
#define STABLEHLO_HELPER_H_

#include "xla/client/xla_computation.h"

namespace mlir {
class ModuleOp;
class MLIRContext;
}  // namespace mlir

namespace torch_xla {
namespace runtime {

std::string hloToStablehlo(const xla::HloModuleProto* proto,
                           bool emit_bytecode);

void ConvertHloToStableHlo(const xla::HloModuleProto* proto,
                           mlir::ModuleOp* mlir_module);

mlir::ModuleOp DeserializeStableHLO(const std::string& bytecode,
                                    mlir::MLIRContext* context);

void convertStableHLOToHLO(mlir::ModuleOp* mlir_module,
                           mlir::MLIRContext* context,
                           xla::HloProto* hlo_proto);

void printHloModuleProto(const xla::HloModuleProto* proto);

}  // namespace runtime
}  // namespace torch_xla

#endif
