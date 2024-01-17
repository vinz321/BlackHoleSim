#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include "object.h"
#include <iostream>


class sphere : public object{
	public:
		__host__ __device__  sphere(vec3_t origin, float radius) :object(origin), radius(radius) { radius_sqr = radius * radius; }

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

	private:

		float radius, radius_sqr;

};
