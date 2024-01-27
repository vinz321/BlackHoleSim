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

__device__ vec3_t march(ray_t& r, cv::cuda::PtrStepSz<vec3_t> hdr, sphere_t* obj_ls, int count, disk_t* disk) {
	vec3_t next_orig;
	vec3_t color;
	vec3_t t = cross(r.dir, norm(obj_ls[0].position - r.orig));
	vec3_t k = norm(t);
	for (int i = 0; i < r.n_steps; i++) {
		next_orig = r.orig + r.delta * r.dir;
		t = cross(r.dir, norm(obj_ls[0].position - r.orig));

		for (int j = 0; j < count; j++) {
			if (is_inside(obj_ls[j+1], next_orig, color)) {
				return color;
			}
		}
		if (is_inside(obj_ls[0], next_orig, color)) {
			return color;
		}
		if (hit_disk(*disk, r.orig, r.dir, r.delta, color)) {
			return color;
		}

		r.dir = norm(rotate(r.dir, k, get_deflection(obj_ls[0], next_orig) * (t * t)));
		r.orig = next_orig;
	}
	/*v = ((atan2f(r.dir.z, r.dir.x) + PI) / (2 * PI)) * 512;
	u = (1 - ((asinf((r.dir.y)) + (PI / 2)) / PI)) * 256;*/
	color = hdr((((asinf((r.dir.y)) + (PI / 2)) / PI)) * 256, (1-(atan2f(r.dir.z, r.dir.x) + PI) / (2 * PI)) * 512);
	return color;
}
