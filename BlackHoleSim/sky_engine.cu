#pragma once
#include "sky_engine.h"
#include <stdlib.h>
#include <iostream>

__host__ __device__ cv::Mat3f read_exr(){
	putenv("OPENCV_IO_ENABLE_OPENEXR=1");
	Mat hdr = imread("hdri/milkyway.jpg", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
	cv::cvtColor(hdr, hdr, cv::COLOR_BGR2RGB);
	/*namedWindow("HDR first", WINDOW_NORMAL);
	imshow("HDR first", hdr);*/
	hdr.convertTo(hdr, 21);
	return hdr;
}