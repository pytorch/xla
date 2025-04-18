// torch_xla/csrc/config.cpp
#include "torch_xla/csrc/config.h"

#include <c10/util/Flags.h>

C10_DEFINE_int(torch_xla_graph_execution_log_level, -1,
               "set torch xla tensor graph execution check level, specify <= 0 "
               "(DISABLED), 1 (WARN), 2 or >=2 (ERROR)");
