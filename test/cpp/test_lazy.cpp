#include <gtest/gtest.h>

#include "tensorflow/compiler/xla/shape.h"
#include "torch/csrc/lazy/core/shape.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla_test.h"

namespace torch_xla {
namespace cpp_test {

class LazyTest : public TorchXlaTest {};

TEST_F(LazyTest, TestXlaShapeToLazy) {
    int64_t dimensions[] = {1, 2};
    bool dynamic_dimensions[] = {false, false};
    absl::Span<const int64_t> xla_dimensions = absl::Span<const int64_t>(dimensions);
    absl::Span<const bool> xla_dynamic_dimensions = absl::Span<const bool>(dynamic_dimensions);
    std::vector<xla::Shape> xla_tuple_shapes = std::vector<xla::Shape>();
    xla::Shape xla_shape = xla::Shape(xla::PrimitiveType::F32, xla_dimensions, xla_dynamic_dimensions, xla_tuple_shapes);

    torch::lazy::Shape lazy_shape = XlaHelpers::ConvertXlaShapeToLazy(xla_shape);
    std::vector<int64_t> lazy_dimensions = xla::util::ToVector<int64_t>(lazy_shape.sizes());
    const c10::optional<std::vector<bool>>& lazy_dynamic_dimensions = lazy_shape.is_symbolic();
    EXPECT_EQ(lazy_shape.scalar_type(), at::ScalarType::Float);
    EXPECT_EQ(lazy_shape.dim(), 2);
    EXPECT_EQ(lazy_dimensions, xla::util::ToVector<int64_t>(xla_dimensions));
    EXPECT_EQ(lazy_dynamic_dimensions.has_value(), false);
}

TEST_F(LazyTest, TestXlaShapeToLazyWithDynamicDimensions) {
    int64_t dimensions[] = {1, 2};
    bool dynamic_dimensions[] = {true, false};
    absl::Span<const int64_t> xla_dimensions = absl::Span<const int64_t>(dimensions);
    absl::Span<const bool> xla_dynamic_dimensions = absl::Span<const bool>(dynamic_dimensions);
    std::vector<xla::Shape> xla_tuple_shapes = std::vector<xla::Shape>();
    xla::Shape xla_shape = xla::Shape(xla::PrimitiveType::F32, xla_dimensions, xla_dynamic_dimensions, xla_tuple_shapes);

    torch::lazy::Shape lazy_shape = XlaHelpers::ConvertXlaShapeToLazy(xla_shape);
    std::vector<int64_t> lazy_dimensions = xla::util::ToVector<int64_t>(lazy_shape.sizes());
    const c10::optional<std::vector<bool>>& lazy_dynamic_dimensions = lazy_shape.is_symbolic();
    EXPECT_EQ(lazy_shape.scalar_type(), at::ScalarType::Float);
    EXPECT_EQ(lazy_shape.dim(), 2);
    EXPECT_EQ(lazy_dimensions, xla::util::ToVector<int64_t>(xla_dimensions));
    EXPECT_EQ(lazy_dynamic_dimensions.has_value(), true);
    EXPECT_EQ(lazy_dynamic_dimensions.value(), std::vector<bool>(std::begin(dynamic_dimensions), std::end(dynamic_dimensions)));
}

}  // namespace cpp_test
}  // namespace torch_xla