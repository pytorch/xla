#include "third_party/xla_client/mhlo_helper.h"
#include "third_party/xla_client/xla_util.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "mlir/IR/Verifier.h"  // from @llvm-project

#include <iostream>

namespace xla {

void hlo_mhlo_hlo_roundtrip_helper(HloModuleProto* proto) {
  mlir::MLIRContext context;
  mlir::ModuleOp mlir_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto status = ConvertHloToMlirHlo(mlir_module, proto, /*import_all_computations=*/false);
  if (!status.ok()) {
    std::cout << "hlo2mhlo not ok" << std::endl;
  }
  if (!mlir::verify(mlir_module).succeeded()) {
    std::cout << "verify not ok" << std::endl;
    TF_ASSIGN_OR_RETURN(auto hlo_module, xla::util::CreateModuleFromProto(*proto));
    std::cout << hlo_module->ToString() << std::endl;
  }
  xla::HloProto hlo_proto;
  mlir::MlirToHloConversionOptions options;
  auto status1 = mlir::ConvertMlirHloToHlo(
    mlir_module, &hlo_proto, /*use_tuple_args=*/false, /*return_tuple=*/false,
    options);
  if (!status1.ok()) {
    std::cout << "mhlo2hlo not ok" << std::endl;
  }
  proto->Swap(hlo_proto.mutable_hlo_module());
}

}