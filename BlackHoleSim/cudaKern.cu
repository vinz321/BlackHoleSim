#include "cudaKern.h"

using namespace cv;
using namespace cv::cuda;

__device__ vec3 get(vec3 *unrolled_matrix,uint x,uint y,uint z, uint size) {
	unsigned long idx = x + y * size + z * size * size;

	return unrolled_matrix[idx];
}

__device__ void set(vec3* unrolled_matrix, vec3 value, uint x, uint y, uint z, uint size) {
	unsigned long idx = x + y * size + z * size * size;
	unrolled_matrix[idx]=value;
}

__global__ void _gravity_field(PtrStepSz<vec3> output) {
	//int bidx = blockIdx.x + blockIdx.y * gridDim.x ;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	//int idx = x + y * gridDim.x * blockDim.x;

	vec3 t = { x / 256.0f,y / 256.0f,0 };

	output(y, x) = t;
}

__global__ void test_kern(vec3 *unrolled) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	vec3 t = {
		x / 256.0f,
		y / 256.0f,
		0
	};
	set(unrolled, t, x, y, 0, 256);
}

Mat calc_gravity_field() {
	vec3* values;
	vec3* values_gpu;
	cudaMalloc(&values_gpu	, 256 * 256 * 3 * sizeof(vec3));
	cudaMallocHost(&values, 256 * 256 * 3 * sizeof(vec3));
	cudaMemset(values_gpu, 0, 256 * 256 * 3 * sizeof(vec3));


	Mat3f test(256,256);
	test.setTo(Vec3f(0, .5f, 1));


	
	GpuMat t_gpu;
	t_gpu.upload(test);

	dim3 grid_size(8,8);
	dim3 block_size(32,32);
	_gravity_field <<<grid_size,block_size>>> (t_gpu);
	//test_kern<<<grid_size,block_size>>>(values_gpu);
	//cudaError_t err = cudaGetLastError();
	//printf("%s", cudaGetErrorString(err));

	//cudaMemcpy(values, values_gpu, 256 * 256 * 3 * sizeof(gpu_vec3), cudaMemcpyDeviceToHost);
	t_gpu.download(test);
	//cudaDeviceSynchronize();

	//test = Mat3f();

	return test;
}

Mat render() {
	camera* cam;
	vec3** d_img;
	vec3** h_img;
	int x = 512;
	int y = 512;

	cudaMallocHost(&h_img, x * y * sizeof(vec3));
	cudaMalloc(&cam, sizeof(camera));
	cudaMalloc(&d_img, x*y*sizeof(camera));

	render <<<1, x*y>>>(d_img, x, y, cam);
	cudaDeviceSynchronize();
	cudaMemcpy(h_img, d_img, x * y * sizeof(vec3), cudaMemcpyDeviceToHost);


}

int add_bi(int a, int b) {
	return a + b;
}