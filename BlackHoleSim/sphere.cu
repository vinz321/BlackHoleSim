#include "sphere.h"

__device__ bool sphere::is_inside (vec3_t point,vec3_t* color) {
	float a, b, c;
	a = orig.x - point.x;
	b = orig.y - point.y;
	c = orig.z - point.z;
	if((a * a + b * b + c * c) > radius_sqr)
		return false;
	
	*color = { 0,0,0 };
	return true;
}