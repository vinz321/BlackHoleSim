#pragma once
#include <stdlib.h>
#include <iostream>
#include "sky_engine.h"

 cv::Mat3f read_exr(){
	//putenv("OPENCV_IO_ENABLE_OPENEXR=1");
	Mat hdr = imread("hdri/milkyway.jpg");
	Mat out_hdri;
	//cv::cvtColor(hdr, out_hdri, cv::COLOR_BGRA2RGB);
	/*namedWindow("HDR first", WINDOW_NORMAL);
	imshow("HDR first", hdr);*/
	hdr.convertTo(hdr, CV_32FC3, 1/255.0f);
	return hdr;
}