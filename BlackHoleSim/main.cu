#include "cuda_profiler_api.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include "tracing.h"
#include <math.h>
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

	Mat3f hdr = read_exr();

	/*cam_pos = vec3_t{ 0,2 * sinf(angle),-2 * cosf(angle) };
	cam_dir = vec3_t{ 0,-sinf(angle),cosf(angle) };
	camera cam(cam_pos, cam_dir, vec3_t{ 0,1,0 }, 60, (float)img_w / img_h);
	cv::Mat m = renderScene(img_w, img_h, &cam);

	cv::cvtColor(m, m, cv::COLOR_RGB2BGR);
	std::cout << angle << std::endl;*/

	//std::cout << m << std::endl;
	//test_kern <<<1, 1 >>> ();
	//std::cout << cudaGetErrorString(cudaGetLastError());

	//cv::imshow("Output", m);
	//cv::Mat3f m = calc_gravity_field();

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	
	std::cout << "Free: " << free << " Total: " << total << std::endl;

	cam_pos = vec3_t{ 0, 0, 0 };
	cam_dir = vec3_t{ 0, 0, 1 };

	//sphere** scene = createScene(0);
	sphere_t* scene = createSceneStruct(0);

	while (true)
	{
		//camera cam(cam_pos, cam_dir, vec3_t{ 0, 1, 0 }, 60, (float)img_w / img_h);
		cam_pos = vec3_t{ 0,2 * sinf(angle),-2 * cosf(angle) };
		cam_dir = vec3_t{ 0,-sinf(angle),cosf(angle) };
		camera cam(cam_pos, cam_dir, vec3_t{ 0,cosf(angle),sinf(angle)}, 60, (float)img_w / img_h);
		cv::Mat3f m = renderScene(img_w, img_h, &cam, angle, hdr, scene);

		cv::cvtColor(m, m, cv::COLOR_RGB2BGR);
		//std::cout << angle << std::endl;
		
		//std::cout << m << std::endl;
		//test_kern <<<1, 1 >>> ();
		//std::cout << cudaGetErrorString(cudaGetLastError());
		
		cv::imshow("Output", m);
		angle += 0.1f;
		cudaMemGetInfo(&free, &total);
		std::cout << "Free: " << free << " Total: " << total << std::endl;
		if ((cv::waitKey(1) & 0xFF) == 'q') {
			cudaProfilerStop();
			break;
		}
	}

	
}