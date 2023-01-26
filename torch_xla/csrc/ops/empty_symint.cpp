//
//namespace torch_xla {
//namespace { // why is this namespace necessary? TODO: try to comment out and see what happens
//
//xla::Shape NodeOutputXLAShape(const torch::lazy::Value& input,
//                           const std::vector<int64_t> upper_bounds,
//                           const absl::InlinedVector<bool, xla::InlineRank()> dynamic_dims) {
//  return xla::ShapeUtil.MakeShape(GetXlaShape(input).element_type(),
//                                  {upper_bounds}, {convertToVecBool(dynamic_dims)});
//}
//
//std::vector<torch::lazy::Value> GetValues(
//  const torch::lazy::Value& input,
//  const std::vector<torch::lazy::NodePtr>& dimensions) {
//  std::vector<torch::lazy::Value> values;
//  values.reserve(dimensions.size()+1);
//  values.push_back(input);
//  for (torch::lazy::NodePtr dim : dimensions) {
//    XLA_CHECK(dim);
//    values.push_back(torch::lazy::Value(dim, 0));
//  }
//  return values;
//}
//
//
//} // namespace
//
//// todo: what's [&]() {}
//EmptySymInt::EmptySymInt(const torch::lazy::Value& input,
//                         const SymIntElements& size_elements)
//    : XlaNode(torch::lazy::OpKind(at::aten::empty),
//              GetValues(input, size_elements.GetSizeNodes()),
//              [&]() {
//                return NodeOutputXLAShape(input, size_elements.GetUpperBounds(),
//                                          size_elements.GetDynamicDims());
//              },
//              /*num_outputs*/1,
//              torch::lazy::MHash(size_elements.GetUpperBounds(),
//                                 convertToVecBool(size_elements.GeDynamicDims()))),
//      upper_bounds_(size_elements.GetUpperBounds()),
//      dynamic_dims_(size_elements.GetDynamicDims()) {}
//
//}
//
//XlaOpVector EmptySymInt::Lower(LoweringContext* loctx) const {
//  xla::XlaOp input = loctx->GetOutputOp(operand(0));
//  std::vector<xla::XlaOp> size_ops;
//  for (int i=1; i < operands().size(); i++) {
//    size_ops.push_back(loctx->GetOutputOp(operand(i)));
//  }
//  xla::XlaOp output = SetDimensionSize(BuildEmpty(input, upper_bounds_),
//                                       size_ops, dynamic_dims_);
//  return ReturnOp(output, loctx);
//}
//
//std::string EmptySymInt::ToString() const {
//  std::stringstream ss;
//  ss << XlaNode::ToString() << ", size(" << absl::StrJoin(upper_bounds_, ", ")
//     << ")"
//     << ", dynamic_dims=(" << absl::StrJoin(dynamic_dims_, ", ") << ")";
//  return ss.str();
//}
//
//} // namespace torch_xla
//
