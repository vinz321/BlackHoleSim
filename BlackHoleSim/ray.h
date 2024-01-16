#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include "sphere.h"
#include "object.h"

class ray {
public:
	__host__ __device__ ray(vec3_t orig, vec3_t dir) :orig(orig), dir(norm(dir)) {};
	__host__ __device__ vec3_t get_orig() { return orig; }
	__host__ __device__ vec3_t get_dir() { return dir; }
	__device__ bool hit_sphere(sphere s) {
		vec3_t orig_diff = (orig - s.get_origin());
		float a = norm(dir) * orig_diff;
		float b = orig_diff * orig_diff - s.get_radius_sqr() * s.get_radius_sqr();
		return (a * a - b) >= 0;
	}

	__device__ bool march(object *obj_ls, int count) {
		vec3_t next_orig;
		for (int i = 0; i < n_seg; i++) {
			next_orig = orig + delta * dir;

		}
	}

private:
	float delta=0.1f;
	int n_seg = 1024;
	vec3_t orig, dir;
};