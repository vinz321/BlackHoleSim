
#include "cudaKern.h"

using namespace cv;
using namespace cv::cuda;

__device__ vec3_t get(vec3_t *unrolled_matrix,uint x,uint y,uint z, uint size) {
	unsigned long idx = x + y * size + z * size * size;

	return unrolled_matrix[idx];
}

__device__ void set(vec3_t* unrolled_matrix, vec3_t value, uint x, uint y, uint z, uint size) {
	unsigned long idx = x + y * size + z * size * size;
	unrolled_matrix[idx]=value;
}

__global__ void _gravity_field(PtrStepSz<vec3_t> output) {
	//int bidx = blockIdx.x + blockIdx.y * gridDim.x ;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	//int idx = x + y * gridDim.x * blockDim.x;

	vec3_t t = { x / 256.0f,y / 256.0f,0 };

	output(y, x) = t;
}

__global__ void test_kern(vec3_t *unrolled) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	vec3_t t = {
		x / 256.0f,
		y / 256.0f,
		0
	};
	set(unrolled, t, x, y, 0, 256);
}

Mat calc_gravity_field() {
	vec3_t* values;
	vec3_t* values_gpu;
	cudaMalloc(&values_gpu	, 256 * 256 * 3 * sizeof(vec3_t));
	cudaMallocHost(&values, 256 * 256 * 3 * sizeof(vec3_t));
	cudaMemset(values_gpu, 0, 256 * 256 * 3 * sizeof(vec3_t));


	Mat3f test(256,256);
	test.setTo(Vec3f(0, .5f, 1));


	
	GpuMat t_gpu;
	t_gpu.upload(test);

	dim3 grid_size(8,8);
	dim3 block_size(32,32);
	_gravity_field <<<grid_size,block_size>>> (t_gpu);


	//test_kern<<<grid_size,block_size>>>(values_gpu);
	cudaError_t err = cudaGetLastError();
	printf("%s", cudaGetErrorString(err));

	//cudaMemcpy(values, values_gpu, 256 * 256 * 3 * sizeof(gpu_vec3), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	t_gpu.download(test);

	//test = Mat3f();

	return test;
}

//void render() {
//	vec3 lfrom{ 0.0f, 0.0f, -1.0f };
//	vec3 lat{ 0.0f, 0.0f, 1.0f };
//	vec3 vup{ 0.0f, 1.0f, 0.0f };
//	float vfov = 60;
//	float aspect = 16 / 9;
//	camera cam(lfrom, lat, vup, vfov, aspect);
//	camera *cam_gpu;
//
//	cudaMalloc(&cam_gpu, sizeof(camera));
//	cudaMemcpyAsync(cam_gpu, cam, sizeof(camera), cudaMemcpyHostToDevice);
//
//	render <<<grid_size, block_size>>>(img, max_x, max_y, cam);
//
//	cudaDeviceSynchronize();
//}

int add_bi(int a, int b) {
	return a + b;
}