#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

typedef struct _vec3 {
	float x;
	float y;
	float z;


} vec3_t;

typedef struct _color {
	float r;
	float g;
	float b;
	float a;
} color_t;


__host__ __device__ vec3_t operator+ (const vec3_t& x, const vec3_t& y);
__host__ __device__ float operator* (const vec3_t& x, const vec3_t& y);
__host__ __device__ vec3_t operator* (const float& y, const vec3_t& x);
__host__ __device__ vec3_t operator- (const vec3_t& x, const vec3_t& y);
__host__ __device__ vec3_t operator/ (const vec3_t& x, const float& y);
__host__ __device__ vec3_t norm(const vec3_t& v);

__host__ __device__ vec3_t cross(const vec3_t& x, const vec3_t& y);
