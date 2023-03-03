#ifndef XLA_MHLO_HELPER_H_
#define XLA_MHLO_HELPER_H_

#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace xla {
    void hlo_mhlo_hlo_roundtrip_helper(HloModuleProto* proto);

    void printHloModuleProto(const HloModuleProto* proto);
}

#endif