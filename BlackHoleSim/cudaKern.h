#include <opencv2/core/cuda.hpp>
#include "tracing.h"

__global__ void _gravity_field(cv::cuda::PtrStepSz<vec3> ptr);

cv::Mat calc_gravity_field();

int add_bi(int a, int b);
