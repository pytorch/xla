#include <gtest/gtest.h>

#include <stdexcept>

#include "cpp_test_util.h"
// #include "tensorflow/compiler/xla/xla_client/async_task.h"

namespace torch_xla {

  class SPMDExecutionTest {
    int replica_count_;
    bool megacore_;

    public:
      SPMDExecutionTest(int replica_count, bool megacore=false) {
        replica_count_ = replica_count;
        megacore_ = megacore;
        if (megacore_) {
          replica_count_ /= 2;
        }
      }

      ~SPMDExecutionTest() {}
    protected:

  }

namespace cpp_test {



}  // namespace cpp_test
}  // namespace torch_xla
