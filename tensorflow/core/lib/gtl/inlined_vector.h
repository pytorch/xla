#pragma once

#include <vector>

namespace tensorflow {

namespace gtl {

template <class T, int v>
using InlinedVector = std::vector<T>;

}  // namespace gtl

}  // namespace tensorflow
