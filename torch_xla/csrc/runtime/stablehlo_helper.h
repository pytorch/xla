#ifndef STABLEHLO_HELPER_H_
#define STABLEHLO_HELPER_H_

#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace torch_xla {
namespace runtime {

std::string hloToStablehloStr(const xla::HloModuleProto* proto);

}  // namespace runtime
}  // namespace torch_xla

#endif
