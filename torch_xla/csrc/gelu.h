#pragma once

namespace torch_xla {

// These constants control the approximation behavior of gelu function.
enum GeluType {
  None,  // Baseline Gelu
  Tanh,  // Tahn Gelu Approximation
  END
};

}  // namespace torch_xla
