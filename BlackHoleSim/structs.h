#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

#define PI 3.1415

#define GRAV_LIGHT_CONST 1.48 //not accurate and is times 10^-27

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

typedef struct _sphere {
	vec3_t position;
	float radius;
	vec3_t color;
	float mass;
}sphere_t;


__host__ __device__ vec3_t operator+ (const vec3_t& x, const vec3_t& y);
__host__ __device__ float operator* (const vec3_t& x, const vec3_t& y);
__host__ __device__ vec3_t operator* (const float& y, const vec3_t& x);
__host__ __device__ vec3_t operator- (const vec3_t& x, const vec3_t& y);
__host__ __device__ vec3_t operator/ (const vec3_t& x, const float& y);
__host__ __device__ vec3_t norm(const vec3_t& v);

__host__ __device__ vec3_t cross(const vec3_t& x, const vec3_t& y);

__host__ __device__ vec3_t rotate(const vec3_t& x, const vec3_t& k, float theta);
