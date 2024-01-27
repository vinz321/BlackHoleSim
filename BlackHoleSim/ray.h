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
	__device__ bool hit_sphere(sphere s);

	__device__ vec3_t march(sphere** obj_ls, sphere blackhole, int count);

	__device__ vec3_t march(sphere_t* obj_ls, sphere_t* blackhole, int count, disk_t* disk);

public:
	float delta=0.02f;
	int n_seg = 256;
	vec3_t orig, dir;
	int u, v;
	cv::cuda::PtrStepSz<vec3_t> hdr;
};

__device__ vec3_t march(ray_t& r, cv::cuda::PtrStepSz<vec3_t> hdr, sphere_t* obj_ls, int count, disk_t* disk);