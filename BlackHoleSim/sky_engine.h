#pragma once

#include "structs.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
__host__ __device__ cv::Mat3f read_exr();