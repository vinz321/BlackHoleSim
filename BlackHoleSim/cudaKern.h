
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>

typedef struct _gpu_vec3 {
	float r;
	float g;
	float b;
} gpu_vec3;

__global__ void _gravity_field(cv::cuda::PtrStepSz<gpu_vec3> ptr);

cv::Mat calc_gravity_field();

int add_bi(int a, int b);
