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

	__device__ bool test_func(vec3_t* test) {
		int a = 1 + 2;

		
	}

	__device__ vec3_t march(sphere **obj_ls, int count) {
		vec3_t next_orig;
		vec3_t color = {1,1,1};


		//bool test;
		for (int i = 0; i < n_seg; i++) {
			next_orig = orig + delta * dir;


			for (int j = 0; j < count; j++) {

				/*float a, b, c;
				a = obj_ls[j]->get_origin().x - next_orig.x;
				b = obj_ls[j]->get_origin().y - next_orig.y;
				c = obj_ls[j]->get_origin().z - next_orig.z;

				if ((a * a + b * b + c * c) > obj_ls[j]->get_radius_sqr())
					test=false;
				else {
					color = vec3_t{ 0,0,0 };
					test = true;
				}*/
				
				if (obj_ls[j]->is_inside(next_orig, &color)) {
				//if(test){
					//*test_pointer = { .1f,.1f,1 };
					test_func(&color);
					goto endLoop;
				}
					
			}
			orig = next_orig;
		}
		endLoop:
		return color;
	}

private:
	float delta=0.1f;
	int n_seg = 1024;
	vec3_t orig, dir;
};