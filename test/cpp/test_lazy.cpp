#include <gtest/gtest.h>

#include "tensorflow/compiler/xla/shape.h"
#include "test/cpp/torch_xla_test.h"
#include "torch/csrc/lazy/core/shape.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace cpp_test {

class LazyTest : public TorchXlaTest {};

TEST_F(LazyTest, TestXlaShapeToLazyWithF64) {
  int64_t dimensions[] = {1};
  bool dynamic_dimensions[] = {false};
  absl::Span<const int64_t> xla_dimensions =
      absl::Span<const int64_t>(dimensions);
  absl::Span<const bool> xla_dynamic_dimensions =
      absl::Span<const bool>(dynamic_dimensions);
  std::vector<xla::Shape> xla_tuple_shapes = std::vector<xla::Shape>();
  xla::Shape xla_shape = xla::Shape(xla::PrimitiveType::F64, xla_dimensions,
                                    xla_dynamic_dimensions, xla_tuple_shapes);

  torch::lazy::Shape lazy_shape = XlaHelpers::ConvertXlaShapeToLazy(xla_shape);
  std::vector<int64_t> lazy_dimensions =
      xla::util::ToVector<int64_t>(lazy_shape.sizes());
  const c10::optional<std::vector<bool>>& lazy_dynamic_dimensions =
      lazy_shape.is_symbolic();
  EXPECT_EQ(lazy_shape.scalar_type(), at::ScalarType::Double);
  EXPECT_EQ(lazy_dimensions, xla::util::ToVector<int64_t>(xla_dimensions));
  EXPECT_EQ(lazy_dynamic_dimensions.has_value(), false);
}

TEST_F(LazyTest, TestXlaShapeToLazyWithPred) {
  int64_t dimensions[] = {1};
  bool dynamic_dimensions[] = {false};
  absl::Span<const int64_t> xla_dimensions =
      absl::Span<const int64_t>(dimensions);
  absl::Span<const bool> xla_dynamic_dimensions =
      absl::Span<const bool>(dynamic_dimensions);
  std::vector<xla::Shape> xla_tuple_shapes = std::vector<xla::Shape>();
  xla::Shape xla_shape = xla::Shape(xla::PrimitiveType::PRED, xla_dimensions,
                                    xla_dynamic_dimensions, xla_tuple_shapes);

  torch::lazy::Shape lazy_shape = XlaHelpers::ConvertXlaShapeToLazy(xla_shape);
  std::vector<int64_t> lazy_dimensions =
      xla::util::ToVector<int64_t>(lazy_shape.sizes());
  const c10::optional<std::vector<bool>>& lazy_dynamic_dimensions =
      lazy_shape.is_symbolic();
  EXPECT_EQ(lazy_shape.scalar_type(), at::ScalarType::Bool);
  EXPECT_EQ(lazy_dimensions, xla::util::ToVector<int64_t>(xla_dimensions));
  EXPECT_EQ(lazy_dynamic_dimensions.has_value(), false);
}

TEST_F(LazyTest, TestXlaShapeToLazyWithU64) {
  int64_t dimensions[] = {1};
  bool dynamic_dimensions[] = {false};
  absl::Span<const int64_t> xla_dimensions =
      absl::Span<const int64_t>(dimensions);
  absl::Span<const bool> xla_dynamic_dimensions =
      absl::Span<const bool>(dynamic_dimensions);
  std::vector<xla::Shape> xla_tuple_shapes = std::vector<xla::Shape>();
  xla::Shape xla_shape = xla::Shape(xla::PrimitiveType::U64, xla_dimensions,
                                    xla_dynamic_dimensions, xla_tuple_shapes);

  torch::lazy::Shape lazy_shape = XlaHelpers::ConvertXlaShapeToLazy(xla_shape);
  std::vector<int64_t> lazy_dimensions =
      xla::util::ToVector<int64_t>(lazy_shape.sizes());
  const c10::optional<std::vector<bool>>& lazy_dynamic_dimensions =
      lazy_shape.is_symbolic();
  EXPECT_EQ(lazy_shape.scalar_type(), at::ScalarType::Long);
  EXPECT_EQ(lazy_dimensions, xla::util::ToVector<int64_t>(xla_dimensions));
  EXPECT_EQ(lazy_dynamic_dimensions.has_value(), false);
}

TEST_F(LazyTest, TestXlaShapeToLazyWithMultipleDimensions) {
  int64_t dimensions[] = {2, 1, 3};
  bool dynamic_dimensions[] = {false, false, false};
  absl::Span<const int64_t> xla_dimensions =
      absl::Span<const int64_t>(dimensions);
  absl::Span<const bool> xla_dynamic_dimensions =
      absl::Span<const bool>(dynamic_dimensions);
  std::vector<xla::Shape> xla_tuple_shapes = std::vector<xla::Shape>();
  xla::Shape xla_shape = xla::Shape(xla::PrimitiveType::F64, xla_dimensions,
                                    xla_dynamic_dimensions, xla_tuple_shapes);

  torch::lazy::Shape lazy_shape = XlaHelpers::ConvertXlaShapeToLazy(xla_shape);
  std::vector<int64_t> lazy_dimensions =
      xla::util::ToVector<int64_t>(lazy_shape.sizes());
  const c10::optional<std::vector<bool>>& lazy_dynamic_dimensions =
      lazy_shape.is_symbolic();
  EXPECT_EQ(lazy_shape.scalar_type(), at::ScalarType::Double);
  EXPECT_EQ(lazy_dimensions, xla::util::ToVector<int64_t>(xla_dimensions));
  EXPECT_EQ(lazy_dynamic_dimensions.has_value(), false);
}

TEST_F(LazyTest, TestXlaShapeToLazyWithDynamicDimensions) {
  int64_t dimensions[] = {2, 1, 3};
  bool dynamic_dimensions[] = {true, false, true};
  absl::Span<const int64_t> xla_dimensions =
      absl::Span<const int64_t>(dimensions);
  absl::Span<const bool> xla_dynamic_dimensions =
      absl::Span<const bool>(dynamic_dimensions);
  std::vector<xla::Shape> xla_tuple_shapes = std::vector<xla::Shape>();
  xla::Shape xla_shape = xla::Shape(xla::PrimitiveType::F64, xla_dimensions,
                                    xla_dynamic_dimensions, xla_tuple_shapes);

  torch::lazy::Shape lazy_shape = XlaHelpers::ConvertXlaShapeToLazy(xla_shape);
  std::vector<int64_t> lazy_dimensions =
      xla::util::ToVector<int64_t>(lazy_shape.sizes());
  const c10::optional<std::vector<bool>>& lazy_dynamic_dimensions =
      lazy_shape.is_symbolic();
  EXPECT_EQ(lazy_shape.scalar_type(), at::ScalarType::Double);
  EXPECT_EQ(lazy_dimensions, xla::util::ToVector<int64_t>(xla_dimensions));
  EXPECT_EQ(lazy_dynamic_dimensions.has_value(), true);
  EXPECT_EQ(lazy_dynamic_dimensions.value(),
            std::vector<bool>(std::begin(dynamic_dimensions),
                              std::end(dynamic_dimensions)));
}

TEST_F(LazyTest, TestXlaShapeToLazyWithUnsupportedPrimitiveType) {
  int64_t dimensions[] = {1};
  bool dynamic_dimensions[] = {false};
  absl::Span<const int64_t> xla_dimensions =
      absl::Span<const int64_t>(dimensions);
  absl::Span<const bool> xla_dynamic_dimensions =
      absl::Span<const bool>(dynamic_dimensions);
  std::vector<xla::Shape> xla_tuple_shapes = std::vector<xla::Shape>();
  xla::Shape xla_shape = xla::Shape(xla::PrimitiveType::TUPLE, xla_dimensions,
                                    xla_dynamic_dimensions, xla_tuple_shapes);

  EXPECT_THROW(XlaHelpers::ConvertXlaShapeToLazy(xla_shape),
               std::runtime_error);
}

}  // namespace cpp_test
}  // namespace torch_xla
