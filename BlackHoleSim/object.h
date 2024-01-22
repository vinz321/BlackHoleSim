#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <stdio.h>

class object {
	public: 
		__host__ __device__ object(vec3_t orig) : orig(orig) {};
		__device__ virtual bool is_inside(vec3_t point, vec3_t& col) { return false; };
		__device__ bool is_inside(vec3_t point0, vec3_t point1, vec3_t* col);
		object* allocGPU();

	protected:
		vec3_t orig;

};
