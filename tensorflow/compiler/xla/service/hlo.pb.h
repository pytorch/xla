#pragma once

#include <string>

#include "lazy_tensors/computation_client/ltc_logging.h"

namespace xla {

class HloModuleProto {
 public:
  std::string SerializeAsString() const {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }
};

}  // namespace xla
