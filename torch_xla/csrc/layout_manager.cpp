#include "torch_xla/csrc/layout_manager.h"

#include <algorithm>
#include <functional>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {
namespace {

xla::Shape MakeShapeWithSortedLayout(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    xla::PrimitiveType type) {
  // Place bigger dimensions on most minor layout locations.
  std::vector<xla::int64> layout =
      xla::util::Iota<xla::int64>(dimensions.size(), dimensions.size() - 1, -1);
  std::sort(layout.begin(), layout.end(), [&](xla::int64 a, xla::int64 b) {
    return dimensions[a] > dimensions[b];
  });
  return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
}

}  // namespace

xla::Shape MakeTorchTensorLayout(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    xla::PrimitiveType type) {
  return xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
}

xla::Shape MakeArrayShapeFromDimensions(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    xla::PrimitiveType type, DeviceType device_type) {
  if (dimensions.size() > 1 && device_type == DeviceType::TPU) {
    return MakeShapeWithSortedLayout(dimensions, type);
  }
  return MakeTorchTensorLayout(dimensions, type);
}

}  // namespace torch_xla
