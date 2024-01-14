#include "ray.h"

__host__ __device__ bool ray::hit_sphere(sphere s) {
	vec3 orig_diff = (orig - s.get_origin());
	float a = dir * orig_diff;
	float b = orig_diff * orig_diff - s.get_radius_sqr() * s.get_radius_sqr();
	return (a * a - b)>=0;
}