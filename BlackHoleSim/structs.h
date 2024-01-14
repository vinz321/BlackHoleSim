#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

typedef struct _vec3 {
	float x;
	float y;
	float z;
} vec3;

typedef struct _color {
	float r;
	float g;
	float b;
	float a;
} color;

__host__ __device__ vec3 norm(const vec3& v) {
	return v / sqrt(v * v);
}

__host__ __device__ vec3 operator+ (const vec3& x, const vec3& y) {
	return vec3{ x.x + y.x,
		x.y + y.y,
		x.z + y.z };
}

__host__ __device__ float operator* (const vec3& x, const vec3& y) {
	return x.x * y.x+
		x.y * y.y+
		x.z * y.z;
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