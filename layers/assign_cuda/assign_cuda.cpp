#include <torch/extension.h>
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")



at::Tensor assign_fcos_cuda(const at::Tensor &label_cls, const at::Tensor &label_box, 
					const int ph, const int pw, const int stride,
					const int size_min, const int size_max);
at::Tensor assign_fcos(const at::Tensor &label_cls, const at::Tensor &label_box, 
					const int ph, const int pw, const int stride,
					const int size_min, const int size_max)
{
	CHECK_CUDA(label_cls);
	CHECK_CUDA(label_box);
	return assign_fcos_cuda(label_cls, label_box, ph, pw, 
								stride, size_min, size_max);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
	m.def("assign_fcos", &assign_fcos, "assign_fcos (CUDA)");
}

