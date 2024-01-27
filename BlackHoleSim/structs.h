#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>


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

typedef struct _disk {
	vec3_t position;
	float radius1, radius2;
	vec3_t color;
	vec3_t normal;
} disk_t;

typedef struct _ray {
	vec3_t orig;
	vec3_t dir;
	float delta;
	int n_steps;
} ray_t;

typedef struct _camera {
	vec3_t origin;
	vec3_t lower_left_corner;
	vec3_t horizontal;
	vec3_t vertical;
}camera_t;


__host__ __device__ vec3_t operator+ (const vec3_t& x, const vec3_t& y);
__host__ __device__ float operator* (const vec3_t& x, const vec3_t& y);
__host__ __device__ vec3_t operator* (const float& y, const vec3_t& x);
__host__ __device__ vec3_t operator- (const vec3_t& x, const vec3_t& y);
__host__ __device__ vec3_t operator/ (const vec3_t& x, const float& y);
__host__ __device__ vec3_t norm(const vec3_t& v);
__host__ __device__ vec3_t cross(const vec3_t& x, const vec3_t& y);
__host__ __device__ vec3_t rotate(const vec3_t& x, const vec3_t& k, float theta);
__host__ __device__ bool hit_disk(disk_t& disk, vec3_t& point, vec3_t& dir, float step_size, vec3_t& color);
