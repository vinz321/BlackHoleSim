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

void render_screen() {

}

int main() {
	
	vec3_t test = { 1,2,3 };
	test = test + test;
	float img_w = 512;
	float img_h = 256;
	float angle = 0;
	vec3_t cam_pos= vec3_t{ 0,2*sinf(angle),-2*cosf(angle)};
	vec3_t cam_dir= vec3_t{ 0,-sinf(angle),cosf(angle)};
	Mat3f hdri_cpu = read_exr();
	cv::resize(hdri_cpu, hdri_cpu, Size(1024, 512));

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

	cudaStream_t mem_stream;
	cudaStreamCreate(&mem_stream);

	cudaEvent_t start, end; 
	
	float time;

	sphere_t* scene = createSceneStruct(0, mem_stream); //BASELINE
	GpuMat hdri;
	hdri.upload(hdri_cpu);

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	
	//for(int i=0;i<3;i++)
	while(true)
	{
		/*if(i==4)
			cudaProfilerStart();*/
		cudaEventRecord(start);
		scene= createSceneStruct(angle, mem_stream); //BASELINE

		cam_pos = vec3_t{ 2*cosf(angle), 2* sinf(angle), -0.25f};
		cam_dir = norm(vec3_t{0,0,0} - cam_pos);
		camera_t cam = make_cam(cam_pos, cam_dir, vec3_t{ 0,0,1}, 60, (float)img_w / img_h);

		createSceneInConstant(2*angle, mem_stream, &cam); //CONSTANT
		cv::Mat m = renderScene(hdri, img_w, img_h, angle, scene, (disk_t *)(scene + 3), &cam); //BASELINE
		//cv::Mat m = renderScene(hdri, img_w, img_h, angle); //SHARED
		//cv::Mat m = renderSceneConst(hdri, img_w, img_h, angle); //CONSTANT

		cudaEventRecord(end);
		
		cv::imshow("Output", m);
		angle += 0.1f;
		
		cudaEventElapsedTime(&time, start, end);
		std::cout << "Time elapsed: " << time << std::endl;
		//cudaMemGetInfo(&free, &total);
		//std::cout << "Free: " << free << " Total: " << total << std::endl;
		if ((cv::waitKey(1) & 0xFF) == 'q') {
			cudaProfilerStop();
			break;
		}
	}
}


//Constant memory 85 ms