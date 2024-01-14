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


__host__ __device__ vec3 operator+ (const vec3& x, const vec3& y);
__host__ __device__ float operator* (const vec3& x, const vec3& y);
__host__ __device__ vec3 operator- (const vec3& x, const vec3& y);
__host__ __device__ vec3 operator/ (const vec3& x, const float& y);
__host__ __device__ vec3 norm(const vec3& v);