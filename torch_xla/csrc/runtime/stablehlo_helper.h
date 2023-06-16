#ifndef STABLEHLO_HELPER_H_
#define STABLEHLO_HELPER_H_

#include "xla/client/xla_computation.h"

namespace torch_xla {
namespace runtime {

std::string hloToStablehlo(const xla::HloModuleProto* proto,
                           bool emit_bytecode);

}  // namespace runtime
}  // namespace torch_xla

#endif
