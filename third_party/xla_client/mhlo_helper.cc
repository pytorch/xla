#include "third_party/xla_client/mhlo_helper.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

namespace xla {

void hlo_mhlo_hlo_roundtrip_helper(HloModuleProto* proto) {
  mlir::MLIRContext context;
  mlir::ModuleOp mlir_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto status = ConvertHloToMlirHlo(mlir_module, proto, /*import_all_computations=*/false);
  xla::HloProto hlo_proto;
  mlir::MlirToHloConversionOptions options;
  auto status1 = mlir::ConvertMlirHloToHlo(
    mlir_module, &hlo_proto, /*use_tuple_args=*/false, /*return_tuple=*/false,
    options);
  proto->Swap(hlo_proto.mutable_hlo_module());
}

}