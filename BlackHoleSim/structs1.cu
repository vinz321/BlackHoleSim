#include "structs.h"

__host__ __device__ vec3 operator+ (const vec3& x, const vec3& y) {
	return vec3{ x.x + y.x,
		x.y + y.y,
		x.z + y.z };
}

__host__ __device__ float operator* (const vec3& x, const vec3& y) {
	return x.x * y.x +
		x.y * y.y +
		x.z * y.z;
}
__host__ __device__ vec3 operator* (const float& y, const vec3& x) {
	return vec3{ x.x * y,x.y * y, x.z * y, };
}

__host__ __device__ vec3 operator- (const vec3& x, const vec3& y) {
	return vec3{ x.x - y.x,
		x.y - y.y,
		x.z - y.z };
}

__host__ __device__ vec3 operator/ (const vec3& x, const float& y) {
	return vec3{ x.x / y,
		x.y / y,
		x.z / y };
}
__host__ __device__ vec3 norm(const vec3& v) {
	return v / sqrtf(v * v);
}

__host__ __device__ vec3 cross(const vec3& x, const vec3& y) {
	return vec3{ x.y * y.z - x.z * y.y, -(x.x * y.z - x.z * y.x), x.x * y.y - x.y * y.x };
}