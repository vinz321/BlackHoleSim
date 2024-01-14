#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "cudaKern.h"

using namespace cv::cuda;

int main() {
	cv::Mat m = calc_gravity_field();
	//std::cout << m << std::endl;
	cv::imshow("Output", m);

	cv::waitKey();
}