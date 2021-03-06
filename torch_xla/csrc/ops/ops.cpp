#include "ops/ops.h"
#include "data_ops.h"
#include "elementwise.h"
#include "helpers.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {
namespace ops {

NodePtr ReluOp(const NodeOperand& input) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_output = BuildRelu(xla_input);
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1) << "Unexpected number of operands";
    return BuildRelu(operands[0]);
  };
  xla::Shape output_shape =
      ir::ops::InferOutputShape({input.node->shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::relu), ir::OpList{input},
                            output_shape, std::move(lower_fn));
}

NodePtr TransposeOp(const NodeOperand& input) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_output = xla::Transpose(xla_input, {1, 0});
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1) << "Unexpected number of operands";
    return xla::Transpose(operands[0], {1, 0});
  };
  xla::Shape output_shape =
      ir::ops::InferOutputShape({input.node->shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::t), ir::OpList{input},
                            output_shape, std::move(lower_fn));
}

NodePtr AddMatMulOp(const NodeOperand& input, const NodeOperand& weight,
                    const NodeOperand& bias, bool use_full_conv_precision) {
  const auto precision_level = use_full_conv_precision
                                   ? xla::PrecisionConfig::HIGHEST
                                   : xla::PrecisionConfig::DEFAULT;
  auto lower_fn = [precision_level](
                      const ir::Node& node,
                      ir::LoweringContext* loctx) -> ir::XlaOpVector {
    XLA_CHECK_EQ(node.operands().size(), 3) << "Unexpected number of operands";
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_weight = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_bias = loctx->GetOutputOp(node.operand(2));
    const auto bias_sizes =
        XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(xla_bias));
    xla::PrecisionConfig precision_config =
        XlaHelpers::BuildPrecisionConfig(precision_level);
    xla::XlaOp xla_dot = xla::Dot(xla_input, xla_weight, &precision_config);
    const auto dot_sizes =
        XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(xla_dot));
    if (bias_sizes != dot_sizes) {
      xla_bias = BuildExpand(xla_bias, dot_sizes);
    }
    xla::XlaOp xla_output = xla_dot + xla_bias;
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2) << "Unexpected number of operands";
    return xla::Dot(operands[0], operands[1]);
  };
  xla::Shape output_shape = ir::ops::InferOutputShape(
      {input.node->shape(), weight.node->shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::addmm),
                            ir::OpList{input, weight, bias}, output_shape,
                            std::move(lower_fn));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
