#include "sphere.h"

__host__ __device__ bool sphere::is_inside(vec3 point) {
	int a, b, c;
	a = origin.x - point.x;
	b = origin.y - point.y;
	c = origin.z - point.z;

	return (a * a + b * b + c * c) <= radius_sqr;
}