#ifndef XLA_TORCH_XLA_CSRC_COMMON_BASE_H_
#define XLA_TORCH_XLA_CSRC_COMMON_BASE_H_

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DISALLOW_COPY_AND_MOVE(className)           \
  className(const className &) = delete;            \
  className &operator=(const className &) = delete; \
  className(className &&) = delete;                 \
  className &operator=(className &&) = delete
#endif

#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif