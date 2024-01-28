#include "ray.h"

//__host__ __device__ bool ray::hit_sphere(sphere s) {
//	vec3 orig_diff = (orig - s.get_origin());
//	float a = norm(dir) * orig_diff;
//	float b = orig_diff * orig_diff - s.get_radius_sqr() * s.get_radius_sqr();
//	return (a * a - b)>=0;
//}

__device__ bool ray::hit_sphere(sphere s) {
	vec3_t orig_diff = (orig - s.get_origin());
	float a = norm(dir) * orig_diff;
	float b = orig_diff * orig_diff - s.get_radius_sqr() * s.get_radius_sqr();
	return (a * a - b) >= 0;
}

__device__ vec3_t ray::march(sphere** obj_ls, sphere blackhole, int count) {
	vec3_t next_orig;
	vec3_t color = hdr(v, u);
	vec3_t t = cross(dir, norm(blackhole.get_origin() - orig));
	vec3_t k = norm(t);
	for (int i = 0; i < n_seg; i++) {
		next_orig = orig + delta * dir;
		t = cross(dir, norm(blackhole.get_origin() - orig));

		for (int j = 0; j < count; j++) {
			if (obj_ls[j]->is_inside(next_orig, color)) {
				return color;
			}
		}
		if (blackhole.is_inside(next_orig, color)) {
			return color;
		}
		dir = rotate(dir, k, blackhole.get_deflection(next_orig, 0.01f) * (t * t));
		orig = next_orig;
	}
	color = hdr(((dir.y + 1) / 2 * 256), ((2 - dir.x) / 2 * 512));
	return color;
}

__device__ vec3_t ray::march(sphere_t* obj_ls, sphere_t* blackhole, int count, disk_t* disk) {
	vec3_t next_orig;
	vec3_t color;
	vec3_t t = cross(dir, norm(blackhole->position - orig));
	vec3_t k = norm(t);
	for (int i = 0; i < n_seg; i++) {
		next_orig = orig + delta * dir;
		t = cross(dir, norm(blackhole->position - orig));

		for (int j = 0; j < count; j++) {
			if (is_inside(obj_ls[j], next_orig, color)) {
				return color;
			}
		}
		if (is_inside(*blackhole, next_orig, color)) {
			return color;
		}
		if (hit_disk(*disk, orig, dir, delta,color)) {
			return color;
		}

		dir = norm(rotate(dir, k, get_deflection(*blackhole, next_orig) * (t * t)));
		orig = next_orig;
	}
	color = hdr(((dir.y + 1) / 2 * 256), ((2 - dir.x) / 2 * 512));
	return color;
	}

__device__ vec3_t march(ray_t& r, cv::cuda::PtrStepSz<vec3_t> hdr, sphere_t* obj_ls, sphere_t* blackhole, int count, disk_t* disk) {
	vec3_t next_orig;
	vec3_t color;
	vec3_t t = cross(r.dir, norm(blackhole->position - r.orig));
	vec3_t k = norm(t);

	#pragma unroll
	for (int i = 0; i < r.n_steps; i++) {
		next_orig = r.orig + r.delta * r.dir;
		t = cross(r.dir, norm(blackhole->position - r.orig));

		for (int j = 0; j < count; j++) {
			if (is_inside(obj_ls[j], next_orig, color)) {
				return color;
			}
		}

		if (is_inside(*blackhole, next_orig, color)) {
			return color;
		}

		if (hit_disk(*disk, r.orig, r.dir, r.delta, color)) {
			return color;
		}

		r.dir = norm(rotate(r.dir, k, get_deflection(*blackhole, next_orig) * (t * t)));
		r.orig = next_orig;
	}
	color = hdr(((r.dir.y + 1) / 2 * 256), ((2 - r.dir.x) / 2 * 512));
	return color;
}
