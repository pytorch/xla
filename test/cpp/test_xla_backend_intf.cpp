#include <vector>

#include "cpp_test_util.h"
#include "torch_xla/csrc/tensor_util.h"
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

TEST(XLABackendTest, TestScalarTransfer) {
  torch::lazy::BackendImplInterface* impl = GetXlaBackendImpl();
  at::Scalar input = at::Scalar(int64_t(1));
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    torch::lazy::BackendDataPtr data =
        impl->MakeComputationDataFromScalar(input, device);
    at::Tensor res = impl->MakeTensorFromComputationData(data, at::kByte);
    AllClose(at::ones({}, at::TensorOptions(at::kByte)), res);
  });
}

TEST(XLABackendTest, TestPlaceholder) {
  torch::lazy::BackendImplInterface* impl = GetXlaBackendImpl();
  torch::lazy::Shape shape(at::kFloat, {10, 10});
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    torch::lazy::BackendDataPtr data =
        impl->CreateDataPlaceholder(device, shape);
    xla::ComputationClient::DataPtr computation_data = UnwrapXlaData(data);
    EXPECT_EQ(computation_data->device(), device.toString());
    EXPECT_EQ(computation_data->shape(),
              MakeXlaShapeFromLazyShape(shape, device));
  });
}

xla::XlaComputation CreateAddComputation(const xla::Shape& shape) {
  xla::XlaBuilder builder("AddComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, shape, "x");
  xla::XlaOp y = xla::Parameter(&builder, 1, shape, "y");
  xla::XlaOp sum = xla::Add(x, y);
  return ConsumeValue(builder.Build());
}

TEST(XLABackendTest, TestE2E) {
  torch::lazy::BackendImplInterface* impl = GetXlaBackendImpl();
  xla::Shape input_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {8, 8});
  at::Tensor one = at::ones({8, 8}, at::TensorOptions(at::kFloat));
  std::vector<at::Tensor> tensors = {one, one};

  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    xla::XlaComputation xla_computation = CreateAddComputation(input_shape);
    torch::lazy::ComputationPtr computation =
        std::make_shared<torch_xla::Computation>(
            "test", std::move(xla_computation), device);
    std::vector<torch::lazy::ComputationPtr> compiled_programs =
        impl->Compile({computation});
    EXPECT_EQ(compiled_programs.size(), 1);

    std::vector<torch::lazy::BackendDataPtr> parameters;
    for (auto& tensor : tensors) {
      parameters.push_back(impl->MakeComputationDataFromTensor(
          tensor, torch::lazy::Shape(tensor.scalar_type(), tensor.sizes()),
          device));
    }
    std::vector<torch::lazy::BackendDataPtr> res =
        impl->ExecuteComputation(compiled_programs[0], parameters, device);
    EXPECT_EQ(res.size(), 1);
    at::Tensor res_tensor =
        impl->MakeTensorFromComputationData(res[0], at::kFloat);
    AllClose(one + one, res_tensor);
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
