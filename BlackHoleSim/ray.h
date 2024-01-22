#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include "sphere.h"
#include "object.h"
#include "test_head.h"
#include "sky_engine.h"

class ray {
public:
	__host__ __device__ ray(vec3_t orig, vec3_t dir, int u, int v, cv::cuda::PtrStepSz<vec3_t> hdr) :orig(orig), dir(norm(dir)), u(u), v(v), hdr(hdr){};
	__host__ __device__ vec3_t get_orig() { return orig; }
	__host__ __device__ vec3_t get_dir() { return dir; }
	__device__ bool hit_sphere(sphere s) {
		vec3_t orig_diff = (orig - s.get_origin());
		float a = norm(dir) * orig_diff;
		float b = orig_diff * orig_diff - s.get_radius_sqr() * s.get_radius_sqr();
		return (a * a - b) >= 0;
	}


	__device__ vec3_t march(sphere **obj_ls, sphere blackhole, int count) {
		vec3_t next_orig;
		vec3_t color = hdr(v, u);
		vec3_t t = cross(dir, norm(blackhole.get_origin() - orig));
		vec3_t k = norm(t);
		for (int i = 0; i < n_seg; i++) {
			next_orig = orig + delta * dir;
			t = cross(dir, norm(blackhole.get_origin() - orig));

			for (int j = 0; j < count; j++) {
				if (obj_ls[j]->is_inside(next_orig, color)) {
					goto endLoop;
				}
			}
			if (blackhole.is_inside(next_orig, color)) {
				goto endLoop;
			}
			dir = rotate(dir, k, blackhole.get_deflection(next_orig, 0.01f) * (t * t));
			orig = next_orig;
		}
		endLoop:
		return color;
	}

private:
	float delta=0.1f;
	int n_seg = 128;
	vec3_t orig, dir;
	int u, v;
	cv::cuda::PtrStepSz<vec3_t> hdr;
};