#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include "cudaKern.h"

using namespace cv::cuda;

int main() {
	cv::Mat frame = calc_gravity_field();
	cv::Mat frame1 = renderize();
	//std::cout << m << std::endl;
	cv::imshow("Output", frame1);

	cv::waitKey();
	printf("Hi guys im dumb CUDA");
	return 0;
}