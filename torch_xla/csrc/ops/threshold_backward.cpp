#include "ops/threshold_backward.h"
#include "elementwise.h"
#include "lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

ThresholdBackward::ThresholdBackward(const NodeOperand& grad_output,
                                     const NodeOperand& input, float threshold)
    : Node(ir::OpKind(at::aten::threshold_backward), {grad_output, input},
           input.node->shape()),
      threshold_(threshold) {}

XlaOpVector ThresholdBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildThreshold(input, grad_output, threshold_, 0);
  return ReturnOp(output, loctx);
}

std::string ThresholdBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", threshold=" << threshold_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
