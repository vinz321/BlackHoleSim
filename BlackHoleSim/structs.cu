#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"

__host__ __device__ vec3_t operator+ (const vec3_t& x, const vec3_t& y) {
	return vec3_t{ x.x + y.x,
		x.y + y.y,

		x.z + y.z };
}

__host__ __device__ float operator* (const vec3_t& x, const vec3_t& y) {
	return x.x * y.x +
		x.y * y.y +
		x.z * y.z;
}
__host__ __device__ vec3_t operator* (const float& y, const vec3_t& x) {
	return vec3_t{ x.x * y,x.y * y, x.z * y, };
}

__host__ __device__ vec3_t operator- (const vec3_t& x, const vec3_t& y) {
	return vec3_t{ x.x - y.x,
		x.y - y.y,
		x.z - y.z };
}

__host__ __device__ vec3_t operator/ (const vec3_t& x, const float& y) {
	return vec3_t{ x.x / y,
		x.y / y,
		x.z / y };
}
__host__ __device__ vec3_t norm(const vec3_t& v) {
	return v / sqrtf(v * v);
}
__host__ __device__ vec3_t cross(const vec3_t& x, const vec3_t& y) {
	return vec3_t{ x.y * y.z - x.z * y.y, -(x.x * y.z - x.z * y.x), x.x * y.y - x.y * y.x };
}

