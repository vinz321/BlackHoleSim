
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <opencv2/core/cuda.hpp>



__global__ void _gravity_field(cv::cuda::PtrStepSz<vec3> ptr);

cv::Mat calc_gravity_field();

int add_bi(int a, int b);
