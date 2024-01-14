#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include "sphere.h"

class ray {
public:
	__host__ __device__ ray(vec3 orig, vec3 dir) :orig(orig), dir(dir) {};
	__host__ __device__ vec3 get_orig() { return orig; }
	__host__ __device__ vec3 get_dir() { return dir; }
	__host__ __device__ bool hit_sphere(sphere s);
private:
	vec3 orig, dir;
};