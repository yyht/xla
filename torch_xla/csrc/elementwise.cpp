#include "elementwise.h"

#include "helpers.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {

xla::XlaOp BuildArithmeticOp(const torch::jit::Node* node,
                             const xla::XlaOp& lhs, const xla::XlaOp& rhs) {
  switch (node->kind()) {
    case at::aten::add: {
      return XlaHelpers::PromotedAdd(lhs, rhs);
    }
    case at::aten::mul: {
      return XlaHelpers::PromotedMul(lhs, rhs);
    }
    case at::aten::sub: {
      return XlaHelpers::PromotedSub(lhs, rhs);
    }
    case at::aten::div: {
      return XlaHelpers::PromotedDiv(lhs, rhs);
    }
    default:
      XLA_ERROR() << "Invalid binary operator kind: " << node->kind();
  }
}

xla::XlaOp BuildComparisonOp(const torch::jit::Node* node,
                             const xla::XlaOp& operand) {
  auto builder = operand.builder();
  xla::Shape operand_shape = XlaHelpers::ShapeOfXlaOp(operand);
  const auto xla_other = XlaHelpers::ScalarValue(
      node->get<at::Scalar>(at::attr::other).value().to<float>(),
      operand_shape.element_type(), builder);
  xla::XlaOp pred;
  switch (node->kind()) {
    case at::aten::gt: {
      pred = xla::Gt(operand, xla_other);
      break;
    }
    default:
      XLA_ERROR() << "Invalid binary operator kind: " << node->kind();
  }
  return xla::ConvertElementType(pred, xla::PrimitiveType::S8);
}

xla::XlaOp BuildThreshold(const xla::XlaOp& input, const xla::XlaOp& output,
                          const float threshold, const float value) {
  auto builder = input.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  const auto input_sizes = XlaHelpers::ShapeSizes(input_shape);
  std::vector<xla::int64> broadcast_sizes(input_sizes.begin(),
                                          input_sizes.end());
  xla::Shape output_shape = XlaHelpers::ShapeOfXlaOp(output);
  const auto xla_threshold = XlaHelpers::ScalarValue<float>(
      threshold, input_shape.element_type(), builder);
  const auto xla_value = XlaHelpers::ScalarValue<float>(
      value, output_shape.element_type(), builder);
  return xla::Select(xla::Gt(input, xla_threshold), output,
                     xla::Broadcast(xla_value, broadcast_sizes));
}

xla::XlaOp BuildRelu(const xla::XlaOp& input) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Max(input, XlaHelpers::ScalarValue<float>(
                             0, input_shape.element_type(), input.builder()));
}

xla::XlaOp BuildTypeAs(const torch::jit::Node* node,
                       const xla::XlaOp& operand) {
  const auto node_outputs = node->outputs();
  XLA_CHECK_EQ(node_outputs.size(), 1);
  const auto output_tensor_type =
      node_outputs[0]->type()->cast<at::TensorType>();
  XLA_CHECK(output_tensor_type);
  const auto target_type = XlaHelpers::MakeXlaPrimitiveType(
      output_tensor_type->scalarType(), /*device=*/nullptr);
  return xla::ConvertElementType(operand, target_type);
}

}  // namespace torch_xla
