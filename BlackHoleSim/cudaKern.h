#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>
#include "tracing.h"
#include "camera.h"

__global__ void _gravity_field(cv::cuda::PtrStepSz<vec3_t> ptr);

cv::Mat calc_gravity_field();

int add_bi(int a, int b);
