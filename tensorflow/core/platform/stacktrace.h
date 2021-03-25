#pragma once

#include <string>

namespace tensorflow {

inline std::string CurrentStackTrace() {
  TF_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace tensorflow
