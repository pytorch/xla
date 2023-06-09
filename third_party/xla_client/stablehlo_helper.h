#ifndef STABLEHLO_HELPER_H_
#define STABLEHLO_HELPER_H_

#include "xla/client/xla_computation.h"

namespace xla {

std::string hloToStablehloStr(const HloModuleProto* proto);

}

#endif
