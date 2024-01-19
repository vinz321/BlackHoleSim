#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"


class object {
	public: 
		__host__ __device__ object(vec3_t orig) : orig(orig) {};
		__device__ bool is_inside(vec3_t* col);

	private:
		vec3_t orig;

};