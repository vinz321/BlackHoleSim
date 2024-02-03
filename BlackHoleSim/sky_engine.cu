#pragma once
#include <stdlib.h>
#include <iostream>
#include "sky_engine.h"

 cv::Mat4f read_exr(){
	//putenv("OPENCV_IO_ENABLE_OPENEXR=1");
	Mat hdr = imread("C:/Users/giovi/Dropbox (Politecnico Di Torino Studenti)/Polito/Secondo anno/GPU programming/Progetto/BlackHoleSim/BlackHoleSim/hdri/milkyway.jpg");

	cv::cvtColor(hdr, hdr, cv::COLOR_BGR2BGRA);	
	hdr.convertTo(hdr, CV_32FC4, 1/255.0f);
	printf("TYPE= %d", hdr.type());
	return hdr;
}