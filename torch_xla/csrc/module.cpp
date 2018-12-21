#include "module.h"
#include "helpers.h"

#include <algorithm>
#include <set>
#include "batch_norm.h"
#include "c10/util/Exception.h"
#include "convolution.h"
#include "cross_replica_reduces.h"
#include "data_ops.h"
#include "elementwise.h"
#include "log_softmax.h"
#include "nll_loss.h"
#include "passes/eval_static_size.h"
#include "passes/remove_in_place_out_param_ops.h"
#include "passes/remove_unused_forward_outputs.h"
#include "passes/replace_in_place_ops.h"
#include "passes/replace_untraced_operators.h"
#include "passes/threshold_backward_peephole.h"
#include "pooling.h"
#include "reduction.h"
#include "size_ops.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch/csrc/jit/passes/canonicalize_ops.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/specialize_undef.h"

namespace torch {
namespace jit {
namespace {

void GatherParameters(std::vector<at::Tensor*>* values,
                      std::vector<bool>* requires_grad,
                      const script::Module& m) {
  for (auto& param : m.get_parameters()) {
    values->push_back(param->slot());
    requires_grad->push_back(!param->is_buffer);
  }
  for (const auto& sub : m.get_modules()) {
    GatherParameters(values, requires_grad, *sub->module);
  }
}

XlaModule::TensorBatchVector CreateResultBatchVector(
    std::vector<std::vector<std::shared_ptr<xla::ComputationClient::Data>>>
        results) {
  XlaModule::TensorBatchVector batch_tensors;
  for (auto& replica_result_components : results) {
    XlaModule::TensorBatchVector::value_type replica_tensors;
    for (auto& replica_data : replica_result_components) {
      replica_tensors.push_back(XLATensor::Create(std::move(replica_data),
                                                  /*requires_grad=*/false));
    }
    batch_tensors.push_back(std::move(replica_tensors));
  }
  return batch_tensors;
}

// Returns the number of real outputs from the forward graph pointed to by
// df_input_vjps. df_input_vjps contains an ordered subset of the full real
// outputs set, followed by an ordered subset of the additional outputs.
size_t InputVjpsRealOutputCount(const Gradient& gradient) {
  size_t real_output_count = 0;
  for (; real_output_count < gradient.df_input_vjps.size();
       ++real_output_count) {
    if (gradient.df_input_vjps[real_output_count] >= gradient.f_real_outputs) {
      break;
    }
  }
  return real_output_count;
}

}  // namespace

XlaModule::XlaModule(const std::shared_ptr<script::Module> module,
                     bool use_full_conv_precision, bool differentiate)
    : use_full_conv_precision_(use_full_conv_precision),
      differentiate_(differentiate),
      script_module_(module) {}

void XlaModule::Initialize(const TensorBatchVector& inputs) {
  if (script_module_ == nullptr) {
    return;
  }

  // Get forward graph.
  const auto forward = script_module_->find_method("forward");
  JIT_ASSERT(forward != nullptr);
  std::shared_ptr<Graph> forward_graph = forward->graph()->copy();
  RunForwardPasses(&forward_graph);

  // Convert model parameters to vector of XLATensors.
  std::vector<at::Tensor*> params_buffers_regather;
  std::vector<bool> param_requires_grad;
  GatherParameters(&params_buffers_regather, &param_requires_grad,
                   *script_module_);
  // The loop below is going to send individual parameters to the different
  // cores. We might need to do something smarter here.
  devices_ = CommonDevicesForReplicas(inputs);
  for (const auto& device : devices_) {
    TensorBatchVector::value_type replica_params;
    TensorBatchVector::value_type optimizable_replica_params;
    for (size_t j = 0; j < params_buffers_regather.size(); ++j) {
      replica_params.push_back(XLATensor::Create(
          autograd::as_variable_ref(*params_buffers_regather[j]), device));
      if (param_requires_grad[j]) {
        optimizable_replica_params.push_back(replica_params.back());
      }
    }
    all_params_.push_back(std::move(replica_params));
    optimizable_params_.push_back(std::move(optimizable_replica_params));
  }
  if (!differentiate_) {
    gradient_.f = forward_graph;
    gradient_.f_real_outputs = forward_graph->outputs().size();
    return;
  }
  // Collect the requires-gradient property making sure all the replica inputs
  // agree on it.
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& replica_inputs = inputs[i];
    if (i == 0) {
      for (const auto& p : replica_inputs) {
        inputs_require_grad_.push_back(p->RequiresGrad());
      }
    } else {
      for (size_t j = 0; j < replica_inputs.size(); ++j) {
        XLA_CHECK(inputs_require_grad_[j] == replica_inputs[j]->RequiresGrad())
            << "Input " << j << " of replica " << i
            << " does not match the requires-grad property";
      }
    }
  }
  inputs_require_grad_.insert(inputs_require_grad_.end(),
                              param_requires_grad.begin(),
                              param_requires_grad.end());

  gradient_ = ComputeGradient(forward_graph);

  TF_VLOG(4) << "Gradient F:\n" << gradient_.f->toString();
  TF_VLOG(4) << "Gradient DF:\n" << gradient_.df->toString();
  // Release the reference to the script module to mark initialization as done.
  script_module_ = nullptr;
}

void XlaModule::RunForwardPasses(std::shared_ptr<Graph>* graph) {
  // Run forward passes.
  CanonicalizeOps(*graph);
  EvalStaticSize(*graph);
  ConstantPropagation(*graph);
  ReplaceUntracedOperators(*graph);
  RemoveInPlaceOutParamOps(*graph);
  ReplaceInPlaceOps(*graph);
  EliminateDeadCode(*graph);
  LowerAllTuples(*graph);
}

Gradient XlaModule::ComputeGradient(const std::shared_ptr<Graph>& graph) {
  // Automatically differentiate the forward graph to get the backward graph.
  // Since differentiation is mutating the graph, do it on a copy.
  std::shared_ptr<Graph> graph_copy = graph->copy();
  Gradient gradient = differentiate(graph_copy);
  // Run the forward passes.
  CanonicalizeOps(gradient.f);
  ConstantPropagation(gradient.f);
  ReplaceUntracedOperators(gradient.f);
  EliminateDeadCode(gradient.f);
  // Run the backward passes.
  specializeUndef(*(gradient.df.get()));
  ConstantPropagation(gradient.df);
  ThresholdBackwardPeephole(gradient.df);
  EliminateDeadCode(gradient.df);
  LowerAllTuples(gradient.df);
  // Run pass on forward and backward graphs that drops outputs that XLA doesn't
  // need.
  RemoveUnusedForwardOutputs(&gradient);
  return gradient;
}

void XlaModule::CheckInitialized() const {
  // script_module_ is null after initialization.
  if (script_module_ != nullptr) {
    AT_ERROR("Module not initialized; did forward method run?");
  }
}

XlaModule::TensorBatchVector XlaModule::forward(
    const TensorBatchVector& inputs) {
  Initialize(inputs);
  if (!backward_input_gradients_.empty()) {
    const auto return_node = gradient_.df->return_node();
    const auto node_inputs = return_node->inputs();
    if (!node_inputs.empty()) {
      return RunFusedTrain(inputs);
    }
  }
  return RunUnfusedForward(inputs);
}

namespace {

class OpByOpContext {
 public:
  std::shared_ptr<XLATensor> GetTensorForInput(const Node* node,
                                               size_t input_index) const {
    const auto input = node->input(input_index);
    auto it = node_tensors_.find(input->unique());
    XLA_CHECK(it != node_tensors_.end())
        << "Input " << input_index << " of " << *node << " not found";
    return it->second;
  }

  std::shared_ptr<XLATensor> GetTensorForId(size_t id) const {
    auto it = node_tensors_.find(id);
    XLA_CHECK(it != node_tensors_.end())
        << "Node with id " << id << " not found";
    return it->second;
  }

  c10::optional<std::shared_ptr<XLATensor>> GetTensorForInputMaybe(
      const Node* node, size_t input_index) const {
    const auto input = node->input(input_index);
    if (undefined_inputs_.find(input->unique()) != undefined_inputs_.end()) {
      return c10::nullopt;
    }
    return GetTensorForId(input->unique());
  }

  void AddTensorForId(const size_t id, std::shared_ptr<XLATensor> tensor) {
    const auto it_ok = node_tensors_.emplace(id, tensor);
    XLA_CHECK(it_ok.second) << "Duplicated tensor id " << id;
  }

  void AddUndefinedInput(size_t index) { undefined_inputs_.insert(index); }

 private:
  std::unordered_map<size_t, std::shared_ptr<XLATensor>> node_tensors_;
  std::unordered_set<size_t> undefined_inputs_;
};

void DispatchOneOp(const Node* node, const std::vector<XLATensor*>& operands,
                   const XlaModule::TensorBatchVector& inputs,
                   OpByOpContext* ctx, xla::XlaBuilder* b,
                   const XlaModule* module) {
  auto xla_computation = b->Build().ValueOrDie();
  xla::Shape result_shape = XlaModule::GetResultShape(xla_computation, inputs);
  const auto computation = XlaGetClient()->Compile(
      std::move(xla_computation), module->GetStringDevices(), &result_shape);
  std::vector<xla::ComputationClient::Data*> arguments;
  for (const auto operand : operands) {
    arguments.push_back(operand->CurrentXlaData().get());
  }
  xla::ComputationClient::ExecuteComputationOptions execute_options;
  const auto result = XlaGetClient()->ExecuteComputation(
      *computation, arguments, computation->devices()[0], execute_options);
  XLA_CHECK_LE(result.size(), node->outputs().size())
      << "Unexpected result size " << result.size();
  for (size_t i = 0; i < result.size(); ++i) {
    ctx->AddTensorForId(node->output(i)->unique(),
                        XLATensor::Create(result[i], false));
  }
}

struct BinaryOpByOpInputs {
  std::shared_ptr<XLATensor> lhs;
  std::shared_ptr<XLATensor> rhs;
  xla::XlaOp xla_lhs;
  xla::XlaOp xla_rhs;
};

c10::optional<xla::XlaOp> GetOpByOpConstOp(
    const size_t id, const std::unordered_map<size_t, Node*>& constant_nodes,
    const std::unordered_map<size_t, xla::Shape>& zero_node_ids,
    xla::XlaBuilder* b) {
  const auto constant_nodes_it = constant_nodes.find(id);
  if (constant_nodes_it != constant_nodes.end()) {
    return GetConstantOp(b, constant_nodes_it->second);
  }
  const auto zero_node_ids_it = zero_node_ids.find(id);
  if (zero_node_ids_it != zero_node_ids.end()) {
    return XlaHelpers::ScalarBroadcast<float>(0, zero_node_ids_it->second, b);
  }
  return c10::nullopt;
}

BinaryOpByOpInputs GetBinaryOpByOpInputs(
    const Node* node, const std::unordered_map<size_t, Node*>& constant_nodes,
    const std::unordered_map<size_t, xla::Shape>& zero_node_ids,
    const OpByOpContext& ctx, xla::XlaBuilder* b) {
  XLA_CHECK_GE(node->inputs().size(), 2);
  const auto xla_lhs_maybe = GetOpByOpConstOp(node->input(0)->unique(),
                                              constant_nodes, zero_node_ids, b);
  const auto xla_rhs_maybe = GetOpByOpConstOp(node->input(1)->unique(),
                                              constant_nodes, zero_node_ids, b);
  const auto lhs = xla_lhs_maybe ? nullptr : ctx.GetTensorForInput(node, 0);
  const auto rhs = xla_rhs_maybe ? nullptr : ctx.GetTensorForInput(node, 1);
  const auto xla_lhs = xla_lhs_maybe
                           ? *xla_lhs_maybe
                           : xla::Parameter(b, 0, lhs->shape(), "param_0");
  size_t second_param_idx = lhs ? 1 : 0;
  const auto xla_rhs =
      xla_rhs_maybe
          ? *xla_rhs_maybe
          : xla::Parameter(b, second_param_idx, rhs->shape(),
                           "param_" + std::to_string(second_param_idx));
  return {lhs, rhs, xla_lhs, xla_rhs};
}

std::vector<XLATensor*> GetBinaryOpByOpTensorOperands(
    const BinaryOpByOpInputs& binary_op_by_op_inputs) {
  std::vector<XLATensor*> operands;
  if (binary_op_by_op_inputs.lhs) {
    operands.push_back(binary_op_by_op_inputs.lhs.get());
  }
  if (binary_op_by_op_inputs.rhs) {
    operands.push_back(binary_op_by_op_inputs.rhs.get());
  }
  return operands;
}

}  // namespace

XlaModule::OpByOpExecutionResult XlaModule::ExecuteOpByOp(
    const TensorBatchVector& inputs,
    const XlaComputationInOut::SizeOpValues& param_size_op_values,
    const std::vector<XlaTranslator::ParameterShape>& param_shapes,
    Graph* graph) {
  const auto graph_inputs = graph->inputs();
  XLA_CHECK_EQ(param_shapes.size(), graph_inputs.size()) << "Graph:\n"
                                                         << graph->toString();
  // TODO(asuhan)
  XLA_CHECK_EQ(inputs.size(), 1)
      << "Op-by-op not supported in replicated mode yet";
  auto nodes = graph->block()->nodes();
  OpByOpContext ctx;
  size_t input_number = 0;
  XlaComputationInOut::SizeOpValues size_op_values;
  std::unordered_map<size_t, xla::Shape> zero_node_ids;
  for (size_t parameter_number = 0; parameter_number < graph_inputs.size();
       ++parameter_number) {
    Value* graph_input = graph_inputs[parameter_number];
    // Seed aten::size tracking info with the values in param_size_op_values.
    const auto size_op_value_it = param_size_op_values.find(parameter_number);
    if (size_op_value_it != param_size_op_values.end()) {
      const auto it_ok = size_op_values.emplace(graph_input->unique(),
                                                size_op_value_it->second);
      XLA_CHECK(it_ok.second)
          << "Duplicated aten::size id" << graph_input->unique();
    }
    if (param_shapes[parameter_number].kind ==
        XlaTranslator::ParameterKind::kZeroInput) {
      const auto it_ok = zero_node_ids.emplace(
          graph_input->unique(), param_shapes[parameter_number].shape);
      XLA_CHECK(it_ok.second);
      continue;
    }
    ctx.AddTensorForId(graph_inputs[parameter_number]->unique(),
                       inputs[0][input_number++]);
  }
  size_t node_idx = 0;
  std::unordered_map<size_t, Node*> constant_nodes;
  for (const auto node : nodes) {
    xla::XlaBuilder b("node_" + std::to_string(node_idx));
    b.set_die_immediately_on_error(true);
    switch (node->kind()) {
      case aten::add:
      case aten::sub:
      case aten::mul: {
        const auto binary_op_by_op_inputs =
            GetBinaryOpByOpInputs(node, constant_nodes, zero_node_ids, ctx, &b);
        xla::PrecisionConfig precision_config =
            XlaHelpers::BuildPrecisionConfig(GetPrecisionConfig());
        auto promoted = XlaHelpers::PromoteValues(
            binary_op_by_op_inputs.xla_lhs, binary_op_by_op_inputs.xla_rhs);
        BuildArithmeticOp(node, promoted.first, promoted.second);
        const auto operands =
            GetBinaryOpByOpTensorOperands(binary_op_by_op_inputs);
        DispatchOneOp(node, operands, inputs, &ctx, &b, this);
        break;
      }
      case aten::convolution:
      case aten::thnn_conv2d_forward: {
        if (node->inputs().size() < 3) {
          AT_ERROR("Unsupported number of inputs for convolution: ",
                   node->inputs().size());
        }

        const auto input = ctx.GetTensorForInput(node, 0);
        const auto kernel = ctx.GetTensorForInput(node, 1);
        const auto bias_maybe = ctx.GetTensorForInputMaybe(node, 3);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        const auto xla_kernel =
            xla::Parameter(&b, 1, kernel->shape(), "param_1");
        if (bias_maybe) {  // bias exists
          const auto bias = *bias_maybe;
          const auto xla_bias = xla::Parameter(&b, 2, bias->shape(), "param_2");
          BuildConvolutionBias(node, xla_input, xla_kernel, xla_bias,
                               GetPrecisionConfig());
          DispatchOneOp(node, {input.get(), kernel.get(), bias.get()}, inputs,
                        &ctx, &b, this);
        } else {
          BuildConvolution(node, xla_input, xla_kernel, GetPrecisionConfig());
          DispatchOneOp(node, {input.get(), kernel.get()}, inputs, &ctx, &b,
                        this);
        }
        break;
      }
      case aten::thnn_conv2d_backward: {
        XLA_CHECK_EQ(node->inputs().size(), 9);
        const auto grad = ctx.GetTensorForInput(node, 0);
        const auto xla_grad = xla::Parameter(&b, 0, grad->shape(), "param_0");
        const auto input = ctx.GetTensorForInput(node, 1);
        const auto xla_input = xla::Parameter(&b, 1, input->shape(), "param_1");
        const auto weight = ctx.GetTensorForInput(node, 2);
        const auto xla_weight =
            xla::Parameter(&b, 2, weight->shape(), "param_2");
        const auto conv2d_grads = BuildConv2dBackward(
            node, xla_grad, xla_input, xla_weight, GetPrecisionConfig());
        XlaHelpers::CreateReturnValue(
            &b, {conv2d_grads.grad_input, conv2d_grads.grad_weight,
                 conv2d_grads.grad_bias});
        DispatchOneOp(node, {grad.get(), input.get(), weight.get()}, inputs,
                      &ctx, &b, this);
        break;
      }
      case aten::t: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        xla::Transpose(xla_input, {1, 0});
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::mm: {
        const auto binary_op_by_op_inputs =
            GetBinaryOpByOpInputs(node, constant_nodes, zero_node_ids, ctx, &b);
        xla::PrecisionConfig precision_config =
            XlaHelpers::BuildPrecisionConfig(GetPrecisionConfig());
        xla::Dot(binary_op_by_op_inputs.xla_lhs, binary_op_by_op_inputs.xla_rhs,
                 &precision_config);
        const auto operands =
            GetBinaryOpByOpTensorOperands(binary_op_by_op_inputs);
        DispatchOneOp(node, operands, inputs, &ctx, &b, this);
        break;
      }
      case aten::max_pool2d_with_indices: {
        XLA_CHECK_GE(node->inputs().size(), 1);
        XLA_CHECK_GE(node->outputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        BuildMaxPool2d(node, xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::max_pool2d_with_indices_backward: {
        XLA_CHECK_EQ(node->inputs().size(), 8);
        const auto out_backprop = ctx.GetTensorForInput(node, 0);
        const auto xla_out_backprop =
            xla::Parameter(&b, 0, out_backprop->shape(), "param_0");
        const auto input = ctx.GetTensorForInput(node, 1);
        const auto xla_input = xla::Parameter(&b, 1, input->shape(), "param_1");
        BuildMaxPool2dBackward(node, xla_out_backprop, xla_input);
        DispatchOneOp(node, {out_backprop.get(), input.get()}, inputs, &ctx, &b,
                      this);
        break;
      }
      case aten::avg_pool2d: {
        XLA_CHECK_GE(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        BuildAvgPool2d(node, xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::avg_pool2d_backward: {
        XLA_CHECK_GE(node->inputs().size(), 2);
        const auto out_backprop = ctx.GetTensorForInput(node, 0);
        const auto xla_grad_output =
            xla::Parameter(&b, 0, out_backprop->shape(), "param_0");
        const auto input = ctx.GetTensorForInput(node, 1);
        const auto xla_input = xla::Parameter(&b, 1, input->shape(), "param_1");
        BuildAvgPool2dBackward(node, xla_grad_output, xla_input);
        DispatchOneOp(node, {out_backprop.get(), input.get()}, inputs, &ctx, &b,
                      this);
        break;
      }
      case aten::neg: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        Neg(xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::tanh: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        Tanh(xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::sigmoid: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        const auto half = XlaHelpers::ScalarValue<float>(
            0.5, input->shape().element_type(), &b);
        xla::XlaOp xla_output = half + half * Tanh(half * xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::relu: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        xla::Shape xla_input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
        xla::Max(xla_input, XlaHelpers::ScalarValue<float>(
                                0, xla_input_shape.element_type(), &b));
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::threshold: {
        XLA_CHECK_EQ(node->inputs().size(), 3);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto output = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        const auto xla_output =
            xla::Parameter(&b, 1, output->shape(), "param_1");
        BuildThreshold(
            node, xla_input, xla_output,
            node->get<at::Scalar>(attr::threshold).value().to<float>(),
            node->get<at::Scalar>(attr::value).value().to<float>(), &b);
        DispatchOneOp(node, {input.get(), output.get()}, inputs, &ctx, &b,
                      this);
        break;
      }
      case aten::threshold_backward: {
        XLA_CHECK_EQ(node->inputs().size(), 3);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto output = ctx.GetTensorForInput(node, 1);
        const auto xla_output =
            xla::Parameter(&b, 0, output->shape(), "param_0");
        const auto xla_input = xla::Parameter(&b, 1, input->shape(), "param_1");
        BuildThreshold(
            node, xla_output, xla_input,
            node->get<at::Scalar>(attr::threshold).value().to<float>(), 0, &b);
        DispatchOneOp(node, {output.get(), input.get()}, inputs, &ctx, &b,
                      this);
        break;
      }
      case aten::log_softmax: {
        XLA_CHECK_EQ(node->inputs().size(), size_t(2));
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        BuildLogSoftmax(node, xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::_log_softmax_backward_data: {
        XLA_CHECK_EQ(node->inputs().size(), 4);
        const auto grad_output = ctx.GetTensorForInput(node, 0);
        const auto xla_grad_output =
            xla::Parameter(&b, 0, grad_output->shape(), "param_0");
        const auto output = ctx.GetTensorForInput(node, 1);
        const auto xla_output =
            xla::Parameter(&b, 1, output->shape(), "param_1");
        BuildLogSoftmaxGrad(node, xla_grad_output, xla_output);
        DispatchOneOp(node, {grad_output.get(), output.get()}, inputs, &ctx, &b,
                      this);
        break;
      }
      case aten::reshape:
      case aten::view: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        BuildView(node, xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::expand: {
        XLA_CHECK_GE(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        BuildExpand(node, xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::stack: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        std::vector<std::shared_ptr<XLATensor>> operands_owned;
        std::vector<xla::XlaOp> xla_operands;
        const auto stack_inputs = InputListAttr(node, node->input(0)->unique());
        for (size_t i = 0; i < stack_inputs.size(); ++i) {
          const auto input = ctx.GetTensorForId(stack_inputs[i]->unique());
          operands_owned.push_back(input);
          xla_operands.push_back(xla::Parameter(&b, i, input->shape(),
                                                "param_" + std::to_string(i)));
        }
        BuildStack(
            node,
            [&xla_operands, stack_inputs](const Value* val) -> xla::XlaOp {
              const auto it =
                  std::find(stack_inputs.begin(), stack_inputs.end(), val);
              XLA_CHECK(it != stack_inputs.end());
              return xla_operands[it - stack_inputs.begin()];
            },
            &b);
        std::vector<XLATensor*> operands;
        for (const auto& operand : operands_owned) {
          operands.push_back(operand.get());
        }
        DispatchOneOp(node, operands, inputs, &ctx, &b, this);
        break;
      }
      case aten::native_batch_norm:
      case aten::batch_norm: {
        XLA_CHECK_EQ(node->inputs().size(), 8);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto weight = ctx.GetTensorForInput(node, 1);
        const auto bias = ctx.GetTensorForInput(node, 2);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        const auto xla_weight =
            xla::Parameter(&b, 1, weight->shape(), "param_1");
        const auto xla_bias = xla::Parameter(&b, 2, bias->shape(), "param_2");
        const auto batch_norm_output =
            BuildBatchNorm(node, xla_input, xla_weight, xla_bias);
        XlaHelpers::CreateReturnValue(
            &b, {batch_norm_output.output, batch_norm_output.save_mean,
                 batch_norm_output.save_invstd_eps});
        DispatchOneOp(node, {input.get(), weight.get(), bias.get()}, inputs,
                      &ctx, &b, this);
        break;
      }
      case aten::native_batch_norm_backward: {
        XLA_CHECK_EQ(node->inputs().size(), 10);
        const auto grad = ctx.GetTensorForInput(node, 0);
        const auto input = ctx.GetTensorForInput(node, 1);
        const auto weight = ctx.GetTensorForInput(node, 2);
        const auto save_mean = ctx.GetTensorForInput(node, 5);
        const auto save_invstd_eps = ctx.GetTensorForInput(node, 6);
        const auto xla_grad = xla::Parameter(&b, 0, grad->shape(), "param_0");
        const auto xla_input = xla::Parameter(&b, 1, input->shape(), "param_1");
        const auto xla_weight =
            xla::Parameter(&b, 2, weight->shape(), "param_2");
        const auto xla_save_mean =
            xla::Parameter(&b, 3, save_mean->shape(), "param_3");
        const auto xla_save_invstd_eps =
            xla::Parameter(&b, 4, save_invstd_eps->shape(), "param_4");
        auto grads = BuildBatchNormBackward(node, xla_grad,  // grad_output
                                            xla_input,       // input
                                            xla_weight,      // weight
                                            xla_save_mean,   // save_mean
                                            xla_save_invstd_eps);  // save_std
        XlaHelpers::CreateReturnValue(
            &b, {grads.grad_input, grads.grad_weight, grads.grad_bias});
        DispatchOneOp(node,
                      {grad.get(), input.get(), weight.get(), save_mean.get(),
                       save_invstd_eps.get()},
                      inputs, &ctx, &b, this);
        break;
      }
      case aten::sum: {
        XLA_CHECK_GE(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        BuildSum(node, xla_input);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case aten::nll_loss: {
        XLA_CHECK_EQ(node->inputs().size(), 5);
        const auto logits = ctx.GetTensorForInput(node, 0);
        const auto xla_logits =
            xla::Parameter(&b, 0, logits->shape(), "param_0");
        const auto labels = ctx.GetTensorForInput(node, 1);
        const auto xla_labels =
            xla::Parameter(&b, 1, labels->shape(), "param_1");
        BuildNllLoss(node, xla_logits, xla_labels);
        DispatchOneOp(node, {logits.get(), labels.get()}, inputs, &ctx, &b,
                      this);
        break;
      }
      case aten::nll_loss_backward: {
        XLA_CHECK_EQ(node->inputs().size(), 7);
        const auto logits = ctx.GetTensorForInput(node, 1);
        const auto xla_logits =
            xla::Parameter(&b, 0, logits->shape(), "param_0");
        const auto labels = ctx.GetTensorForInput(node, 2);
        const auto xla_labels =
            xla::Parameter(&b, 1, labels->shape(), "param_1");
        BuildNllLossBackward(node, xla_logits, xla_labels);
        DispatchOneOp(node, {logits.get(), labels.get()}, inputs, &ctx, &b,
                      this);
        break;
      }
      case aten::size: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        std::vector<xla::int64> size_op_result;
        BuildSize(node, xla_input, &size_op_result);
        const auto it_ok =
            size_op_values.emplace(node->output(0)->unique(), size_op_result);
        XLA_CHECK(it_ok.second)
            << "Duplicated aten::size id: " << node->output(0)->uniqueName();
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      case prim::Constant: {
        const auto it_ok =
            constant_nodes.emplace(node->output(0)->unique(), node);
        XLA_CHECK(it_ok.second) << "Duplicated prim::Constant id: "
                                << node->output(0)->uniqueName();
        break;
      }
      case prim::ListConstruct: {
        break;
      }
      case prim::Undefined: {
        ctx.AddUndefinedInput(node->output(0)->unique());
        break;
      }
      case prim::SumToSize: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        const auto input = ctx.GetTensorForInput(node, 0);
        const auto xla_input = xla::Parameter(&b, 0, input->shape(), "param_0");
        BuildSumToSize(node, xla_input, size_op_values);
        DispatchOneOp(node, {input.get()}, inputs, &ctx, &b, this);
        break;
      }
      default:
        AT_ERROR("Unsupported operator: ", node->kind().toQualString());
    }
    ++node_idx;
  }
  const auto return_node = graph->return_node();
  const auto node_inputs = return_node->inputs();
  // TODO: tighten the id check for returned tuples.
  if (return_node->kind() != prim::Return || node_inputs.empty()) {
    AT_ERROR("Unexpected end of graph");
  }
  TensorBatchVector::value_type returned_tuple;
  XlaComputationInOut::SizeOpValues ret_size_op_values;
  for (size_t return_input_idx = 0; return_input_idx < node_inputs.size();
       ++return_input_idx) {
    const auto return_input = node_inputs[return_input_idx];
    const auto it = size_op_values.find(return_input->unique());
    if (it != size_op_values.end()) {
      const auto it_ok =
          ret_size_op_values.emplace(return_input_idx, it->second);
      XLA_CHECK(it_ok.second)
          << "Duplicated return component index " << return_input_idx;
    }
    returned_tuple.push_back(
        ctx.GetTensorForInput(return_node, return_input_idx));
  }
  return {{returned_tuple}, ret_size_op_values};
}

void XlaModule::SetInputGradientsForFusion(std::vector<at::Tensor> gradients) {
  if (xla::sys_util::GetEnvInt("XLA_OP_BY_OP", 0)) {
    AT_ERROR("Not supported for op-by-op mode yet");
  }
  backward_input_gradients_ = std::move(gradients);
}

void XlaModule::backward(const TensorBatchVector& grad_outputs) {
  JIT_ASSERTM(differentiate_,
              "Calling backward() on a module with differentiate not set");
  CheckInitialized();

  if (!backward_input_gradients_.empty()) {
    // We already have the gradients from the fused computation, just set the
    // gradients for input and parameters.
    ApplyGradients(grad_inputs_, inputs_, optimizable_params_,
                   inputs_require_grad_, *gradient_.df);
    return;
  }
  // Tensors could have pending in-place operations, apply them first to reset
  // their parent module and thus invalidate the gradients we set aside from the
  // fused computation.
  FlushTensorsOperations();

  // NOTE: The order of the input parameters passed to the BuildComputation()
  // call to build the backward computation is critical, as they have to match
  // the sequence of the graph->inputs() vector. Before the gradients passed in
  // from the backward() call, then then zeroed virtual inputs, and then the
  // captured inputs/outputs.
  TensorBatchVector raw_grad_outputs;
  std::vector<bool> zero_input;
  for (size_t i = 0; i < grad_outputs.size(); ++i) {
    TensorBatchVector::value_type replica_raw_grad_outputs;
    for (auto p : grad_outputs[i]) {
      replica_raw_grad_outputs.push_back(p);
      if (i == 0) {
        zero_input.push_back(false);
      }
    }
    const auto& replica_captured_outputs = captured_outputs_[i];
    const auto input_vjps_real_outputs = InputVjpsRealOutputCount(gradient_);
    XLA_CHECK_EQ(input_vjps_real_outputs, replica_raw_grad_outputs.size());
    for (size_t input_vjp_idx = input_vjps_real_outputs;
         input_vjp_idx < gradient_.df_input_vjps.size(); ++input_vjp_idx) {
      const auto raw_output_index = gradient_.df_input_vjps[input_vjp_idx];
      // The index in gradient_.df_input_vjps points inside all outputs list,
      // both real and captured. Skip the real output count to get the captured
      // output index.
      XLA_CHECK_GE(raw_output_index, input_vjps_real_outputs);
      XLA_CHECK_LT(raw_output_index - input_vjps_real_outputs,
                   replica_captured_outputs.size());
      auto p =
          replica_captured_outputs[raw_output_index - input_vjps_real_outputs];
      replica_raw_grad_outputs.push_back(p);
      if (i == 0) {
        zero_input.push_back(true);
      }
    }
    for (auto p : captured_inputs_outputs_[i]) {
      replica_raw_grad_outputs.push_back(p);
      if (i == 0) {
        zero_input.push_back(false);
      }
    }
    raw_grad_outputs.push_back(std::move(replica_raw_grad_outputs));
  }
  const bool op_by_op = xla::sys_util::GetEnvInt("XLA_OP_BY_OP", 0);
  // The shape for all the replicas are the same, so use replica[0] for
  // building the shapes vector for the BuildComputation() call.
  const auto& replica_raw_grad_outputs = raw_grad_outputs.front();
  std::vector<XlaTranslator::ParameterShape> backward_shapes;
  for (size_t j = 0; j < replica_raw_grad_outputs.size(); ++j) {
    XlaTranslator::ParameterKind kind =
        zero_input[j] ? XlaTranslator::ParameterKind::kZeroInput
                      : XlaTranslator::ParameterKind::kGraphInput;
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        replica_raw_grad_outputs[j]->shape(), kind));
  }
  // If backward graph is not compiled, compile it.
  if (backward_computation_ == nullptr && !op_by_op) {
    XlaTranslator xla_bwd_impl(gradient_.df, GetPrecisionConfig());
    xla::XlaComputation computation =
        xla_bwd_impl
            .BuildComputation("XlaBackward", backward_shapes,
                              backward_size_op_values_,
                              GetBackwardBuildOptions(inputs_.size()))
            .computation;
    xla::Shape result_shape = GetResultShape(computation, grad_outputs);
    backward_computation_ = XlaGetClient()->Compile(
        std::move(computation), GetStringDevices(), &result_shape);
  }
  // Collect the computation client data vector.
  DataBatchVector raw_grad_outputs_data =
      GetDataBatchVector(raw_grad_outputs, &zero_input);

  const auto op_by_op_result =
      op_by_op
          ? ExecuteOpByOp(GetTensorBatchVector(raw_grad_outputs, &zero_input),
                          backward_size_op_values_, backward_shapes,
                          gradient_.df.get())
          : XlaModule::OpByOpExecutionResult{};
  TensorBatchVector grad_inputs =
      op_by_op ? op_by_op_result.tensors
               : Execute(*backward_computation_, raw_grad_outputs_data);

  ApplyGradients(grad_inputs, inputs_, optimizable_params_,
                 inputs_require_grad_, *gradient_.df);
  // Release handles to saved / captured inputs and outputs.
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();
}

void XlaModule::ApplyGradients(const TensorBatchVector& grad_inputs,
                               const TensorBatchVector& inputs,
                               const TensorBatchVector& optimizable_params,
                               const std::vector<bool>& inputs_require_grad,
                               const Graph& df) {
  size_t inputs_require_grad_count =
      std::count(inputs_require_grad.begin(), inputs_require_grad.end(), true);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& replica_grad_inputs = grad_inputs[i];
    auto& replica_inputs = inputs[i];
    auto& replica_optimizable_params = optimizable_params[i];
    XLA_CHECK_EQ(replica_grad_inputs.size(), inputs_require_grad_count)
        << "Graph:\n"
        << df.toString();
    size_t grad_index = 0;
    for (size_t j = 0; j < replica_inputs.size(); j++) {
      if (inputs_require_grad[j]) {
        replica_inputs[j]->SetGradient(replica_grad_inputs[grad_index]);
        ++grad_index;
      }
    }
    for (size_t j = 0; j < replica_optimizable_params.size(); j++) {
      replica_optimizable_params[j]->SetGradient(
          replica_grad_inputs[grad_index]);
      ++grad_index;
    }
  }
}

XlaModule::TensorBatchVector XlaModule::RunFusedTrain(
    const TensorBatchVector& inputs) {
  Initialize(inputs);

  TensorBatchVector inputs_params_buffers = PrepareForwardInput(inputs);
  DataBatchVector inputs_params_buffers_data =
      GetDataBatchVector(inputs_params_buffers, /*zero_input=*/nullptr);
  if (forward_computation_ == nullptr) {
    // Shapes are going to be the same for all replicas, so use the ones of the
    // first replica here.
    const TensorBatchVector::value_type& replica_inputs =
        inputs_params_buffers.front();
    std::vector<XlaTranslator::ParameterShape> forward_shapes;
    for (size_t i = 0; i < replica_inputs.size(); ++i) {
      forward_shapes.push_back(XlaTranslator::ParameterShape(
          replica_inputs[i]->shape(),
          XlaTranslator::ParameterKind::kGraphInput));
    }
    xla::XlaComputation computation =
        BuildFusedTrainComputation(forward_shapes);
    xla::Shape result_shape = GetResultShape(computation, inputs);
    forward_computation_ = XlaGetClient()->Compile(
        std::move(computation), GetStringDevices(), &result_shape);
  }

  TensorBatchVector result_components =
      Execute(*forward_computation_, inputs_params_buffers_data);

  // First gradient_.f_real_outputs are the forward outputs returned to user
  // code.
  XLA_CHECK_LE(gradient_.f_real_outputs, result_components.front().size());
  grad_inputs_.clear();
  TensorBatchVector forward_result;
  for (auto& replica_result_components : result_components) {
    TensorBatchVector::value_type replica_forward_result;
    TensorBatchVector::value_type replica_grad_inputs;
    for (size_t j = 0; j < gradient_.f_real_outputs; ++j) {
      replica_forward_result.push_back(replica_result_components[j]);
    }
    for (size_t j = gradient_.f_real_outputs;
         j < replica_result_components.size(); ++j) {
      replica_grad_inputs.push_back(replica_result_components[j]);
    }
    forward_result.push_back(std::move(replica_forward_result));
    grad_inputs_.push_back(std::move(replica_grad_inputs));
  }
  return forward_result;
}

const XlaModule::TensorBatchVector& XlaModule::parameters() {
  CheckInitialized();
  return optimizable_params_;
}

const XlaModule::TensorBatchVector& XlaModule::parameters_buffers() {
  CheckInitialized();
  return all_params_;
}

xla::PrecisionConfig::Precision XlaModule::GetPrecisionConfig() const {
  return use_full_conv_precision_ ? xla::PrecisionConfig::HIGHEST
                                  : xla::PrecisionConfig::DEFAULT;
}

xla::XlaComputation XlaModule::BuildFusedTrainComputation(
    const std::vector<XlaTranslator::ParameterShape>& forward_shapes) {
  XlaTranslator xla_fwd_impl(gradient_.f, GetPrecisionConfig());
  xla::XlaBuilder b("XlaFusedComputation");
  // Build the forward pass program without compiling it, the backward pass
  // needs to be called before finalizing it.
  auto computation_in_outs = xla_fwd_impl.BuildComputationProgram(
      forward_shapes, backward_size_op_values_, &b);
  // Take the XLA outputs from the forward pass and set them for the backward
  // call in the same order the standalone, unfused version takes its arguments.
  XLA_CHECK(!computation_in_outs.outputs.empty());
  XLA_CHECK_EQ(gradient_.f_real_outputs, backward_input_gradients_.size());
  std::vector<xla::XlaOp> captured_inputs_outputs;
  for (auto i : gradient_.df_input_captured_inputs) {
    captured_inputs_outputs.push_back(computation_in_outs.inputs[i]);
  }
  for (auto i : gradient_.df_input_captured_outputs) {
    captured_inputs_outputs.push_back(computation_in_outs.outputs[i]);
  }
  backward_size_op_values_ = SetBackwardSizeOpValues(
      computation_in_outs.ret_size_op_values, gradient_);
  // NOTE: The order of the input parameters passed to the BuildComputation()
  // call to build the backward computation is critical, as they have to match
  // the sequence of the graph->inputs() vector. Before the gradients passed in
  // by the user, then then zeroed virtual inputs, and then the captured
  // inputs/outputs.
  std::vector<XlaTranslator::ParameterShape> backward_shapes;
  std::vector<xla::XlaOp> backward_operands;
  for (size_t i = 0; i < backward_input_gradients_.size(); ++i) {
    xla::Literal literal =
        GetTensorLiteral(backward_input_gradients_[i], /*shape=*/nullptr);
    xla::XlaOp gradient_op = xla::ConstantLiteral(&b, literal);
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(gradient_op),
        XlaTranslator::ParameterKind::kGraphInput));
    backward_operands.push_back(gradient_op);
  }
  for (size_t input_vjp_idx = backward_input_gradients_.size();
       input_vjp_idx < gradient_.df_input_vjps.size(); ++input_vjp_idx) {
    const auto raw_output_index = gradient_.df_input_vjps[input_vjp_idx];
    XLA_CHECK_LT(raw_output_index, computation_in_outs.outputs.size());
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(computation_in_outs.outputs[raw_output_index]),
        XlaTranslator::ParameterKind::kZeroInput));
  }
  for (auto p : captured_inputs_outputs) {
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(p),
        XlaTranslator::ParameterKind::kGraphInput));
    backward_operands.push_back(p);
  }
  // The arguments are set up correctly, call into the backward computation.
  XlaTranslator xla_bwd_impl(gradient_.df, GetPrecisionConfig());
  auto backward_computation =
      xla_bwd_impl
          .BuildComputation("XlaBackward", backward_shapes,
                            backward_size_op_values_,
                            GetBackwardBuildOptions(inputs_.size()))
          .computation;
  xla::XlaOp backward_op =
      xla::Call(&b, backward_computation, backward_operands);

  // Return the real outputs of the forward, followed by the outputs of the
  // backward.
  std::vector<xla::XlaOp> returned_outputs;
  for (size_t i = 0; i < gradient_.f_real_outputs; ++i) {
    returned_outputs.push_back(computation_in_outs.outputs[i]);
  }
  xla::Shape backward_shape = XlaHelpers::ShapeOfXlaOp(backward_op);
  if (xla::ShapeUtil::IsTuple(backward_shape)) {
    for (xla::int64 i = 0;
         i < xla::ShapeUtil::TupleElementCount(backward_shape); ++i) {
      returned_outputs.push_back(xla::GetTupleElement(backward_op, i));
    }
  } else if (!xla::ShapeUtil::IsEmptyTuple(backward_shape)) {
    returned_outputs.push_back(backward_op);
  }
  XlaHelpers::CreateReturnValue(&b, returned_outputs);

  xla::XlaComputation computation = b.Build().ValueOrDie();
  TF_VLOG(5)
      << "Fused computation:\n"
      << xla::xrt_util::GetComputationHloText(computation).ConsumeValueOrDie();
  return computation;
}

XlaModule::TensorBatchVector XlaModule::RunUnfusedForward(
    const TensorBatchVector& inputs) {
  TensorBatchVector inputs_params_buffers = PrepareForwardInput(inputs);
  DataBatchVector inputs_params_buffers_data =
      GetDataBatchVector(inputs_params_buffers, /*zero_input=*/nullptr);

  // Lazy-convert forward graph to XlaComputation.
  const bool op_by_op = xla::sys_util::GetEnvInt("XLA_OP_BY_OP", 0);
  // Shapes are going to be the same for all replicas, so use the ones of the
  // first replica here.
  std::vector<XlaTranslator::ParameterShape> forward_shapes;
  for (auto p : inputs_params_buffers.front()) {
    forward_shapes.push_back(XlaTranslator::ParameterShape(
        p->shape(), XlaTranslator::ParameterKind::kGraphInput));
  }
  if (forward_computation_ == nullptr && !op_by_op) {
    XlaTranslator xla_fwd_impl(gradient_.f, GetPrecisionConfig());
    auto forward_translation_result = xla_fwd_impl.BuildComputation(
        "XlaForward", forward_shapes, backward_size_op_values_);
    backward_size_op_values_ = SetBackwardSizeOpValues(
        forward_translation_result.ret_size_op_values, gradient_);

    xla::Shape result_shape =
        GetResultShape(forward_translation_result.computation, inputs);
    forward_computation_ = XlaGetClient()->Compile(
        std::move(forward_translation_result.computation), GetStringDevices(),
        &result_shape);
  }

  const auto op_by_op_result =
      op_by_op ? ExecuteOpByOp(PrepareForwardInput(inputs), {}, forward_shapes,
                               gradient_.f.get())
               : XlaModule::OpByOpExecutionResult{};
  TensorBatchVector raw_outputs =
      op_by_op ? op_by_op_result.tensors
               : Execute(*forward_computation_, inputs_params_buffers_data);
  if (op_by_op) {
    backward_size_op_values_ =
        SetBackwardSizeOpValues(op_by_op_result.ret_size_op_values, gradient_);
  }

  TensorBatchVector outputs;
  for (size_t i = 0; i < raw_outputs.size(); ++i) {
    auto& replica_raw_outputs = raw_outputs[i];
    TensorBatchVector::value_type replica_outputs;
    for (size_t j = 0; j < gradient_.f_real_outputs; j++) {
      replica_outputs.push_back(replica_raw_outputs[j]);
    }
    outputs.push_back(std::move(replica_outputs));

    TensorBatchVector::value_type replica_captured_outputs;
    for (size_t j = gradient_.f_real_outputs; j < replica_raw_outputs.size();
         j++) {
      replica_captured_outputs.push_back(replica_raw_outputs[j]);
    }
    captured_outputs_.push_back(std::move(replica_captured_outputs));

    auto& replica_inputs_params_buffers = inputs_params_buffers[i];
    TensorBatchVector::value_type replica_captured_inputs_outputs;
    for (auto j : gradient_.df_input_captured_inputs) {
      replica_captured_inputs_outputs.push_back(
          replica_inputs_params_buffers[j]);
    }
    for (auto j : gradient_.df_input_captured_outputs) {
      replica_captured_inputs_outputs.push_back(replica_raw_outputs[j]);
    }
    captured_inputs_outputs_.push_back(
        std::move(replica_captured_inputs_outputs));
  }
  return outputs;
}

XlaModule::TensorBatchVector XlaModule::PrepareForwardInput(
    const TensorBatchVector& inputs) {
  FlushTensorsOperations();
  // Clear the previous forward's captured vectors.
  // This is needed in case backward is not yet run, but two forward calls were
  // made.
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();

  if (inputs_.empty()) {
    inputs_ = inputs;
  } else {
    ReferenceNewTensorData(inputs, &inputs_);
  }

  TensorBatchVector inputs_params_buffers;
  XLA_CHECK_EQ(inputs_.size(), all_params_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    TensorBatchVector::value_type replica_inputs_params_buffers;
    for (auto& p : inputs_[i]) {
      replica_inputs_params_buffers.push_back(p);
    }
    for (auto& p : all_params_[i]) {
      replica_inputs_params_buffers.push_back(p);
    }
    inputs_params_buffers.push_back(std::move(replica_inputs_params_buffers));
  }
  return inputs_params_buffers;
}

std::vector<std::string> XlaModule::GetStringDevices() const {
  std::vector<std::string> devices(devices_.size());
  for (size_t i = 0; i < devices_.size(); ++i) {
    devices[i] = devices_[i].ToString();
  }
  return devices;
}

XlaModule::TensorBatchVector XlaModule::Execute(
    const xla::ComputationClient::Computation& computation,
    const DataBatchVector& inputs) {
  std::vector<std::vector<std::shared_ptr<xla::ComputationClient::Data>>>
      exec_results;
  if (inputs.size() == 1) {
    xla::ComputationClient::ExecuteComputationOptions options;
    exec_results.push_back(XlaGetClient()->ExecuteComputation(
        computation, inputs.front(), computation.devices()[0], options));
  } else {
    xla::ComputationClient::ExecuteReplicatedOptions options;
    exec_results = XlaGetClient()->ExecuteReplicated(
        computation, inputs, computation.devices(), options);
  }
  return CreateResultBatchVector(std::move(exec_results));
}

XlaTranslator::BuildOptions XlaModule::GetBackwardBuildOptions(
    size_t num_replicas) {
  XlaTranslator::BuildOptions options;
  if (num_replicas > 1) {
    options.output_transform = [num_replicas](const xla::XlaOp& op, size_t) {
      return BuildCrossReplicaSum(op, num_replicas);
    };
  }
  return options;
}

void XlaModule::FlushTensorsOperations() {
  // We might have to do something smarter here, as we are syncing even tensors
  // which are not part of the traning loop. Nothing happens, but if we want to
  // fuse the sync operation with the forward+backward+optimizer, we need to
  // have a path leading to the same XLA computation.
  std::vector<std::shared_ptr<XLATensor>> tensors = XLATensor::GetLiveTensors();
  XLATensor::ApplyPendingGraph(tensors, &apply_context_);
}

void XlaModule::ReferenceNewTensorData(const TensorBatchVector& source,
                                       TensorBatchVector* dest) {
  XLA_CHECK_EQ(source.size(), dest->size());
  for (size_t i = 0; i < source.size(); ++i) {
    const TensorBatchVector::value_type& replica_source = source[i];
    TensorBatchVector::value_type* replica_dest = &(*dest)[i];
    XLA_CHECK_EQ(replica_source.size(), replica_dest->size());
    for (size_t j = 0; j < replica_source.size(); ++j) {
      (*replica_dest)[j]->ReferenceDataFrom(*replica_source[j]);
    }
  }
}

XlaComputationInOut::SizeOpValues XlaModule::SetBackwardSizeOpValues(
    const XlaComputationInOut::SizeOpValues& ret_size_op_values,
    const Gradient& gradient) {
  size_t backward_input_idx = 0;
  XlaComputationInOut::SizeOpValues backward_size_op_values;
  for (const auto out_idx : gradient.df_input_vjps) {
    const auto ret_size_op_value_it = ret_size_op_values.find(out_idx);
    if (ret_size_op_value_it != ret_size_op_values.end()) {
      const auto it_ok = backward_size_op_values.emplace(
          backward_input_idx, ret_size_op_value_it->second);
      XLA_CHECK(it_ok.second)
          << "Duplicated backward_input_idx: " << backward_input_idx;
    }
    ++backward_input_idx;
  }
  backward_input_idx += gradient.df_input_captured_inputs.size();
  for (const auto out_idx : gradient.df_input_captured_outputs) {
    const auto ret_size_op_value_it = ret_size_op_values.find(out_idx);
    if (ret_size_op_value_it != ret_size_op_values.end()) {
      const auto it_ok = backward_size_op_values.emplace(
          backward_input_idx, ret_size_op_value_it->second);
      XLA_CHECK(it_ok.second)
          << "Duplicated backward_input_idx: " << backward_input_idx;
    }
    ++backward_input_idx;
  }
  return backward_size_op_values;
}

XlaModule::DataBatchVector XlaModule::GetDataBatchVector(
    const TensorBatchVector& inputs, const std::vector<bool>* zero_input) {
  DataBatchVector inputs_data;
  for (auto& replica_inputs : inputs) {
    DataBatchVector::value_type replica_inputs_data;
    for (size_t j = 0; j < replica_inputs.size(); ++j) {
      if (zero_input == nullptr || !zero_input->at(j)) {
        replica_inputs_data.push_back(replica_inputs[j]->GetXlaData().get());
      }
    }
    inputs_data.push_back(std::move(replica_inputs_data));
  }
  return inputs_data;
}

XlaModule::TensorBatchVector XlaModule::GetTensorBatchVector(
    const TensorBatchVector& inputs, const std::vector<bool>* zero_input) {
  TensorBatchVector non_zero_inputs;
  for (auto& replica_inputs : inputs) {
    TensorBatchVector::value_type replica_non_zero_inputs;
    for (size_t j = 0; j < replica_inputs.size(); ++j) {
      if (zero_input == nullptr || !zero_input->at(j)) {
        replica_non_zero_inputs.push_back(replica_inputs[j]);
      }
    }
    non_zero_inputs.push_back(std::move(replica_non_zero_inputs));
  }
  return non_zero_inputs;
}

std::vector<XLATensor::Device> XlaModule::CommonDevicesForReplicas(
    const TensorBatchVector& inputs) {
  std::vector<XLATensor::Device> devices;
  std::set<XLATensor::Device> unique_devices;
  for (auto& replica_inputs : inputs) {
    devices.push_back(XLATensor::CommonDeviceForTensors(replica_inputs));
    XLA_CHECK(unique_devices.insert(devices.back()).second)
        << "Duplicated device in different replicas: "
        << devices.back().ToString();
  }
  return devices;
}

xla::Shape XlaModule::GetResultShape(const xla::XlaComputation& computation,
                                     const TensorBatchVector& input_tensors) {
  auto devices = CommonDevicesForReplicas(input_tensors);
  const auto program_shape = computation.GetProgramShape().ValueOrDie();
  const auto result_shape = program_shape.result();
  return MakeShapeWithDeviceLayout(result_shape, devices.front().hw_type);
}

}  // namespace jit
}  // namespace torch
