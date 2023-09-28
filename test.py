import torch
import torch_xla
from torch_xla.core import xla_model as xm
from typing import Tuple, Type, Callable, Union, List
from torch_xla import tf_saved_model_integration

device = xm.xla_device()

from torchvision.models import resnet18
model = resnet18()
model.to(device)
input = torch.rand(10, 3, 224, 224).to(device)
torch_xla._XLAC._xla_mark_dynamic(input, 0)
result = model(input)
hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([result])
print(hlo_content)
print(xm.get_stablehlo([result]))
export_filename = "resnet18_saved_model"

### The folllowing line crashes with
## 2023-09-20 00:15:55.270486: F ./torch_xla/csrc/runtime/debug_macros.h:20] Non-OK-status: status.status() status: UNIMPLEMENTED: CustomCall "stablehlo.dynamic_broadcast_in_dim" is not supported to have a dynamic dimension
## *** Begin stack trace ***
##         tsl::CurrentStackTrace[abi:cxx11]()
##         std::unique_ptr<xla::PjRtLoadedExecutable, std::default_delete<xla::PjRtLoadedExecutable> > ConsumeValue<std::unique_ptr<xla::PjRtLoadedExecutable, std::default_delete<xla::PjRtLoadedExecutable> > >(absl::lts_20230125::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable, std::default_delete<xla::PjRtLoadedExecutable> > >&&)
##         torch_xla::runtime::PjRtComputationClient::Compile(std::vector<torch_xla::runtime::ComputationClient::CompileInstance, std::allocator<torch_xla::runtime::ComputationClient::CompileInstance> >)
##         torch_xla::XLAGraphExecutor::Compile(std::vector<c10::intrusive_ptr<torch_xla::XLATensor, c10::detail::intrusive_target_default_null_type<torch_xla::XLATensor> >, std::allocator<c10::intrusive_ptr<torch_xla::XLATensor, c10::detail::intrusive_target_default_null_type<torch_xla::XLATensor> > > > const&, absl::lts_20230125::Span<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const>, torch::lazy::LazyGraphExecutor::SyncTensorCollection const&, torch::lazy::LazyGraphExecutor::PostOrderData*, std::vector<torch::lazy::Value, std::allocator<torch::lazy::Value> > const&)
##         torch_xla::XLAGraphExecutor::SyncTensorsGraphInternal(std::vector<c10::intrusive_ptr<torch_xla::XLATensor, c10::detail::intrusive_target_default_null_type<torch_xla::XLATensor> >, std::allocator<c10::intrusive_ptr<torch_xla::XLATensor, c10::detail::intrusive_target_default_null_type<torch_xla::XLATensor> > > >*, ab
## sl::lts_20230125::Span<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const>, torch::lazy::LazyGraphExecutor::SyncTensorsConfig const&, bool)
##         torch_xla::XLAGraphExecutor::SyncTensorsGraph(std::vector<c10::intrusive_ptr<torch_xla::XLATensor, c10::detail::intrusive_target_default_null_type<torch_xla::XLATensor> >, std::allocator<c10::intrusive_ptr<torch_xla::XLATensor, c10::detail::intrusive_target_default_null_type<torch_xla::XLATensor> > > >*, absl::lts_
## 20230125::Span<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const>, bool, bool, bool)
##         torch_xla::XLAGraphExecutor::SyncLiveTensorsGraph(torch::lazy::BackendDevice const*, c10::ArrayRef<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, bool)

#tf_saved_model_integration.save_torch_module_as_tf_saved_model(model, (input,), export_filename)



####multiply(same axis dynamic)##a = torch.tensor([[1, 2], [2, 4]], device = device)##torch_xla._XLAC._xla_mark_dynamic(a, 0)##b = torch.tensor([[1, 2], [2, 4]], device = device)##torch_xla._XLAC._xla_mark_dynamic(b, 0)##c = a * b##hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])##print(hlo_content)##print(xm.get_stablehlo([c]))


# ## multiply (same axis dynamic)
# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a * b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# ## multiply (all axes dynamic)
# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# torch_xla._XLAC._xla_mark_dynamic(a, 1)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# torch_xla._XLAC._xla_mark_dynamic(b, 1)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a * b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# ## multiply (possible to infer static shapes)
# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 1)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a * b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# # ## multiply (implicit broadcast)
# a = torch.randn((10,1)).to(device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.randn((5)).to(device=device)
# torch_xla._XLAC._xla_set_tag(b, 'tag_of_b')
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# c = a * b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a + b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))

# a = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# torch_xla._XLAC._xla_mark_dynamic(a, 1)
# b = torch.tensor([[1,2],[2,4]], device=device)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# torch_xla._XLAC._xla_mark_dynamic(b, 1)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_b')
# c = a + b
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([c])
# print(hlo_content)
# print(xm.get_stablehlo([c]))


# a = torch.randn((5,10)).to(device=device)
# torch_xla._XLAC._xla_set_tag(a, 'tag_of_a')
# torch_xla._XLAC._xla_mark_dynamic(a, 0)
# b = torch.randn((10,5)).to(device=device)
# torch_xla._XLAC._xla_set_tag(b, 'tag_of_b')
# torch_xla._XLAC._xla_mark_dynamic(b, 1)
# torch_xla._XLAC._xla_mark_dynamic(b, 0)
# c = torch.randn((5,5)).to(device=device)
# torch_xla._XLAC._xla_set_tag(c, 'tag_of_c')
# torch_xla._XLAC._xla_mark_dynamic(c, 0)
# d = a @ b * c
# hlo_content = torch_xla._XLAC._get_xla_tensors_hlo([d])
# print(hlo_content)
# print(xm.get_stablehlo([d]))
