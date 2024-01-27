#pragma once
#include <stdlib.h>
#include <iostream>
#include "sky_engine.h"

 cv::Mat3f hdriread(int img_w, int img_h){
	Mat hdr = imread("C:/Users/giovi/Dropbox (Politecnico Di Torino Studenti)/Polito/Secondo anno/GPU programming/Progetto/BlackHoleSim/BlackHoleSim/hdri/milkyway.jpg");
	hdr.convertTo(hdr, CV_32FC3, 1/255.0f);
	cv::resize(hdr, hdr, Size(img_w, img_h));

	return hdr;
}