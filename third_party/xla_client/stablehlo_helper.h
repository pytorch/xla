#ifndef XLA_MHLO_HELPER_H_
#define XLA_MHLO_HELPER_H_

#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace xla {

std::string hlo_to_stablehlo_str(const HloModuleProto* proto);

}

#endif
