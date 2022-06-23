#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {

xla::Shape AbsOutputShape(const torch::lazy::Value& input);

xla::Shape AcosOutputShape(const torch::lazy::Value& input);

xla::Shape AcoshOutputShape(const torch::lazy::Value& input);

xla::Shape AsinOutputShape(const torch::lazy::Value& input);

xla::Shape AsinhOutputShape(const torch::lazy::Value& input);

xla::Shape AtanOutputShape(const torch::lazy::Value& input);

xla::Shape AtanhOutputShape(const torch::lazy::Value& input);

xla::Shape CosOutputShape(const torch::lazy::Value& input);

xla::Shape CoshOutputShape(const torch::lazy::Value& input);

xla::Shape ErfOutputShape(const torch::lazy::Value& input);

xla::Shape ErfcOutputShape(const torch::lazy::Value& input);

xla::Shape ErfinvOutputShape(const torch::lazy::Value& input);

xla::Shape ExpOutputShape(const torch::lazy::Value& input);

xla::Shape FloorOutputShape(const torch::lazy::Value& input);

xla::Shape InverseOutputShape(const torch::lazy::Value& input);

xla::Shape LogdetOutputShape(const torch::lazy::Value& input);

xla::Shape MaximumOutputShape(const torch::lazy::Value& input,
                              const torch::lazy::Value& other);

xla::Shape MinimumOutputShape(const torch::lazy::Value& input,
                              const torch::lazy::Value& other);

xla::Shape ReciprocalOutputShape(const torch::lazy::Value& input);

xla::Shape SgnOutputShape(const torch::lazy::Value& input);

xla::Shape SignOutputShape(const torch::lazy::Value& input);

xla::Shape SinOutputShape(const torch::lazy::Value& input);

xla::Shape SinhOutputShape(const torch::lazy::Value& input);

/* Blocked on https://github.com/pytorch/xla/issues/3596 */
// xla::Shape SlogdetOutputShape(const torch::lazy::Value& input);

xla::Shape TanOutputShape(const torch::lazy::Value& input);

}  // namespace torch_xla
