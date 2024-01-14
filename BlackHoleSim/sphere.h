#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include <iostream>


class sphere {
	public:
		__host__ __device__  sphere(vec3 origin, float radius) :origin(origin), radius(radius) { radius_sqr = radius * radius; }

		__host__ __device__ bool is_inside(vec3 point) ;

		

		__host__ __device__ float get_radius() { return radius; }
		__host__ __device__ float get_radius_sqr() { return radius_sqr; }
		__host__ __device__ vec3 get_origin() { return origin; }

	private:
		vec3 origin;
		float radius, radius_sqr;

};
