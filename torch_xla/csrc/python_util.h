#pragma once

#include <string>
#include <vector>

#include <c10/util/Optional.h>

namespace torch_xla {

struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};

c10::optional<SourceLocation> GetPythonFrameTop();

std::vector<SourceLocation> GetPythonFrames();

}  // namespace torch_xla
