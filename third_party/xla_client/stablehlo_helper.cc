#include "third_party/xla_client/stablehlo_helper.h"

#include <iostream>

#include "mlir/IR/Verifier.h"       // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/xla_util.h"

namespace xla {

static std::string getHloModuleStr(const HloModuleProto* proto) {
  auto hlo_module = xla::util::CreateModuleFromProto(*proto);
  return hlo_module.value()->ToString();
}

static std::string getMlirModuleStr(mlir::ModuleOp& mlir_module) {
  std::string txt_mlir_module;
  llvm::raw_string_ostream os{txt_mlir_module};
  mlir_module.print(os);
  return txt_mlir_module;
}

static absl::Status hloToMhloHelper(const HloModuleProto* proto,
                                    mlir::ModuleOp* mlir_module) {
  auto status = xla::ConvertHloToMlirHlo(*mlir_module, proto,
                                         /*import_all_computations=*/false);
  if (!status.ok()) {
    return status;
  }
  if (!mlir::verify(*mlir_module).succeeded()) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "MHLO Module from HLO -> MHLO conversion is not legal.");
  }
  return absl::OkStatus();
}

static absl::Status mhloToStablehloHelper(mlir::ModuleOp* mlir_module,
                                          mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(*mlir_module))) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "StableHLO Module from MHLO -> StableHLO conversion is not leagal.");
  }
  return absl::OkStatus();
  ;
}

std::string hloToStablehloStr(const HloModuleProto* proto) {
  mlir::MLIRContext context;
  mlir::ModuleOp mlir_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  static const std::string err_msg =
      "Please open a github issue to PyTorch/XLA.\nOriginal HLO dump:\n";
  auto status = hloToMhloHelper(proto, &mlir_module);
  XLA_CHECK(status.ok()) << "HLO -> MHLO conversion failed.\n"
                         << status.message() << err_msg
                         << getHloModuleStr(proto);
  status = mhloToStablehloHelper(&mlir_module, &context);
  XLA_CHECK(status.ok()) << "MHLO -> StableHLO conversion failed.\n"
                         << status.message() << err_msg
                         << getHloModuleStr(proto);
  return getMlirModuleStr(mlir_module);
}

}  // namespace xla
