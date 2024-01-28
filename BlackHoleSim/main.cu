#include "cuda_profiler_api.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include "tracing.h"
#include <math.h>
#include "structs.h"
#include "sky_engine.h"

//#include "cudaKern.h"

using namespace cv::cuda;

__global__ void test_kern() {
	vec3_t test = { 1,2,3 };
	vec3_t test2 = test + test;
}

int main() {
	cudaProfilerStart();
	vec3_t test = { 1,2,3 };
	test = test + test;

	std::cout << sizeof(object)<< std::endl;

	std::cout << sizeof(sphere) << std::endl;

	std::cout << sizeof(object*) << std::endl;
	float img_w = 512;
	float img_h = 256;
	float angle = 0;
	vec3_t cam_pos= vec3_t{ 0,2*sinf(angle),-2*cosf(angle)};
	vec3_t cam_dir= vec3_t{ 0,-sinf(angle),cosf(angle)};

	Mat3f hdr = hdriread(img_w, img_h);

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	
	std::cout << "Free: " << free << " Total: " << total << std::endl;

	cam_pos = vec3_t{ 0, 0, 0 };
	cam_dir = vec3_t{ 0, 0, 1 };

	while (true)
	{
		sphere_t* scene = createSceneStruct(2*angle);
		cam_pos = vec3_t{ 2*cosf(angle), 2* sinf(angle), -0.5f};
		//cam_dir = vec3_t{ -sinf(angle),-sinf(PI / 2 * 0.95f)*cosf(angle),cosf(PI / 2 * 0.95f)};

		cam_dir = norm(vec3_t{0,0,0} - cam_pos);
		camera_t cam = make_cam(cam_pos, cam_dir, vec3_t{ 0,0,1}, 80, (float)img_w / img_h);
		cv::Mat m = renderScene(img_w, img_h, &cam, angle, hdr, scene, (disk_t*)(scene + 3));
		cv::imshow("Output", m);
		angle += 0.1f;

		if ((cv::waitKey(1) & 0xFF) == 'q') {
			break;
		}
	}

	return 0;
	
}