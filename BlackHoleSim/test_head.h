#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs.h"

class test_head {

	public:
		__host__ __device__ void test_func(vec3_t* col) {
			int a = 1 + 2;
			*col = { 1,0,0 };

		}
};