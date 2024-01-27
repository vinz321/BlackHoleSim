#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include "object.h"
#include <iostream>



class sphere : public object{
	public:
		__host__ __device__  sphere(vec3_t origin, float radius) :object(origin), radius(radius) { radius_sqr = radius * radius; }
		__host__ __device__  sphere(vec3_t origin, float radius, vec3_t col) : object(origin), radius(radius), col(col) { radius_sqr = radius * radius; }
		//__host__ __device__  sphere(vec3_t origin, float radius, float mass) : object(origin), radius(radius), mass(mass) { radius_sqr = radius * radius; }

		__device__ bool is_inside(vec3_t point, vec3_t& col) override;


		__host__ __device__ float get_radius() { return radius; }
		__host__ __device__ float get_radius_sqr() { return radius_sqr; }
		__host__ __device__ vec3_t get_origin() { return orig; }
		sphere* allocGPU()
		{
			sphere* gpu;
			cudaMallocManaged(&gpu, sizeof(sphere));
			//cudaMemcpy(gpu, this, sizeof(sphere), cudaMemcpyHostToDevice);

			*gpu = *this;

			return gpu;
		} 

		__device__ float get_deflection(vec3_t point, float mass) {
			vec3_t dist_vec = point - orig;
			float dist = sqrt(dist_vec * dist_vec);

			return (GRAV_LIGHT_CONST * mass)/ dist;
		}

	private:
		vec3_t col = { 1,0,0 };
		float radius, radius_sqr;

};

 __device__ bool is_inside(sphere_t &sphere, vec3_t point, vec3_t& col);

__host__ __device__ float get_deflection(sphere_t &sphere, vec3_t point);