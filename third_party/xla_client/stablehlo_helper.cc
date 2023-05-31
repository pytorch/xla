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

static void hlo_mhlo_helper(const HloModuleProto* proto,
                            mlir::ModuleOp* mlir_module) {
  auto status = ConvertHloToMlirHlo(*mlir_module, proto,
                                    /*import_all_computations*/ false);
  XLA_CHECK(status.ok()) << "error in HLO -> MHLO conversion.";
  XLA_CHECK(mlir::verify(*mlir_module).succeeded())
      << "mhlo from hlo2mhlo verify not ok.";
}

static void mhlo_stablehlo_helper(mlir::ModuleOp* mlir_module,
                                  mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  XLA_CHECK(mlir::succeeded(pm.run(*mlir_module)))
      << "mhlo to stablehlo not ok";
}

std::string hlo_to_stablehlo_str(const HloModuleProto* proto) {
  mlir::MLIRContext context;
  mlir::ModuleOp mlir_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  hlo_mhlo_helper(proto, &mlir_module);
  mhlo_stablehlo_helper(&mlir_module, &context);
  std::string txt_mlir_module;
  llvm::raw_string_ostream os{txt_mlir_module};
  mlir_module.print(os);
  return txt_mlir_module;
}

}  // namespace xla
