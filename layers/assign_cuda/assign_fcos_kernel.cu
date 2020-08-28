#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#define DivUp(x,y) (int)ceil((float)x/y)



__global__ void assign_fcos_kernel(const long *label_cls, const float *label_box, 
	float *output, const int ph, const int pw, const int stride, 
	const int size_min, const int size_max, const int n)
{
	int phw_i = blockIdx.x*512 + threadIdx.x;
    if (phw_i >= ph*pw) return;
	int ph_i = phw_i / pw;
	int pw_i = phw_i % pw;
	float center_y = ph_i * stride;
    float center_x = pw_i * stride;
    
	float pred_cls=-1, pred_ymin, pred_xmin, pred_ymax, pred_xmax, pred_area=999999999;
    float pred_idx=-1, pred_ctr=0;
    
	for (int i=0; i<n; i++) {

		float cls  = (float)label_cls[phw_i];
		float ymin = label_box[phw_i*4+0];
		float xmin = label_box[phw_i*4+1];
		float ymax = label_box[phw_i*4+2];
		float xmax = label_box[phw_i*4+3];
		float cy = (ymin + ymax) / 2.0;
		float cx = (xmin + xmax) / 2.0;
		float ch = ymax - ymin + 1;
		float cw = xmax - xmin + 1;

		float oy = cy - center_y;
		float ox = cx - center_x;
        float bxa = ch*cw;
        
		float top = center_y - ymin;
		float bottom = ymax - center_y;
		float left = center_x - xmin;
		float right = xmax - center_x;
		float max_tlbr = max(top, max(left, max(bottom, right)));

		if ((max_tlbr >= size_min) && (max_tlbr <= size_max) && (cls > 0) 
			&& (top > 0) && (bottom > 0) && (left > 0) && (right > 0)
			&& (fabs(oy) < (float)stride) && (fabs(ox) < (float)stride) && (bxa <= pred_area)) {
			pred_cls  = cls;
			pred_idx = (float)i;
			pred_ymin = ymin;
			pred_xmin = xmin;
			pred_ymax = ymax;
			pred_xmax = xmax;
			pred_area = bxa;
			pred_ctr = sqrt(min(left, right)*min(top, bottom)
								/max(left, right)/max(top, bottom));
		}
    }
    
	if (pred_cls > 0) {
		output[phw_i*7 + 0] = pred_cls;
		output[phw_i*7 + 1] = pred_ymin;
		output[phw_i*7 + 2] = pred_xmin;
		output[phw_i*7 + 3] = pred_ymax;
		output[phw_i*7 + 4] = pred_xmax;
		output[phw_i*7 + 5] = pred_ctr;
		output[phw_i*7 + 6] = pred_idx;
	}
}
at::Tensor assign_fcos_cuda(const at::Tensor &label_cls, const at::Tensor &label_box, 
					const int ph, const int pw, const int stride,
					const int size_min, const int size_max)
{
	/*
	*** GPU ~= 6.1

	Param:
	label_cls:  L(ph*pw, n)     0:bg, 1~:fg, 0pad
	label_box:  F(ph*pw, n, 4)  ymin, xmin, ymax, xmax, 0:pad

	ph = 129
	pw = 129
	stride = 8
	size_min = 1
	size_max = 64

	Return:
	target_cls:  L(ph*pw)           -1:ign, 0:bg, 1~:fg
	target_reg:  F(ph*pw, 4)        ymin, xmin, ymax, xmax
	target_ctr:  F(ph*pw)
	target_idx:  L(ph*pw)
    -> F(ph*pw, 1 + 4 + 1 + 1)
	*/
	const int n = label_cls.size(0);
	auto output = at::zeros({ph*pw, 7}, label_box.options());
	if (output.numel() == 0) {
		THCudaCheck(cudaGetLastError());
		return output;
	}
	dim3 grid(DivUp(ph*pw, 512)), block(512);
	assign_fcos_kernel<<<grid, block>>>(
		label_cls.contiguous().data<long>(),
		label_box.contiguous().data<float>(),
		output.contiguous().data<float>(),
		ph, pw, stride, size_min, size_max, n);
	THCudaCheck(cudaGetLastError());
	return output;
}
