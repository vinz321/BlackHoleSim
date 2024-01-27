#include "sphere.h"

__device__ bool sphere::is_inside (vec3_t point,vec3_t& color) {
	float a, b, c;
	a = orig.x - point.x;
	b = orig.y - point.y;
	c = orig.z - point.z;
	if((a * a + b * b + c * c) > radius_sqr)
		return false;
	
	color = this->col;
	return true;
}

 __device__ bool is_inside(sphere_t& sphere, vec3_t point, vec3_t &col){
	float a, b, c;
	a = sphere.position.x - point.x;
	b = sphere.position.y - point.y;
	c = sphere.position.z - point.z;
	if ((a * a + b * b + c * c) > sphere.radius * sphere.radius)
		return false;

	col = sphere.color;
	return true;
}

__host__ __device__ float get_deflection(sphere_t& sphere, vec3_t point) {
	vec3_t dist_vec = point - sphere.position;
	float dist = sqrt(dist_vec * dist_vec);

	return (GRAV_LIGHT_CONST * sphere.mass) / dist;
}