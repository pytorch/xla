#include "cpp_test_util.h"
#include "torch_xla/csrc/xla_backend_impl.h"

namespace torch_xla {
namespace cpp_test {

TEST(XLABackendTest, TestTensorTransfer) {
  torch::lazy::BackendImplInterface* impl = GetXlaBackendImpl();
  at::Tensor input = at::randint(std::numeric_limits<uint8_t>::min(),
                                 std::numeric_limits<uint8_t>::max(), {2, 2},
                                 at::TensorOptions(at::kByte));
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    torch::lazy::BackendDataPtr data = impl->MakeComputationDataFromTensor(
        input, torch::lazy::Shape(input.scalar_type(), input.sizes()), device);
    at::Tensor res = impl->MakeTensorFromComputationData(data, at::kByte);
    AllClose(input, res);
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
