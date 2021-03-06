#include "cpp_test_util.h"

#include <string>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {
namespace cpp_test {

at::Tensor ToTensor(XLATensor& xla_tensor) {
  at::Tensor xtensor = xla_tensor.ToTensor();
  if (xtensor.is_variable()) {
    xtensor = torch::autograd::as_variable_ref(xtensor).data();
  }
  return xtensor;
}

bool EqualValues(at::Tensor a, at::Tensor b) {
  at::ScalarType atype = a.scalar_type();
  at::ScalarType btype = b.scalar_type();
  if (atype != btype) {
    a = a.toType(btype);
  }
  return a.equal(b);
}

void ForEachDevice(const std::function<void(const Device&)>& devfn) {
  std::string default_device =
      xla::ComputationClient::Get()->GetDefaultDevice();
  devfn(Device(default_device));
}

}  // namespace cpp_test
}  // namespace torch_xla
