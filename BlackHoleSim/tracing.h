#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ray.h"
#include "sphere.h"
#include <opencv2/core/cuda.hpp>

#include "structs.h"
#include "camera.h"

__device__ vec3_t color(ray r);

__global__ void render(cv::cuda::PtrStepSz<vec3_t> img, int max_x, int max_y, camera *cam, sphere** scene, int count);

cv::Mat3f renderScene(int img_w, int img_h, camera* cam, float& angle, Mat3f &hdr);