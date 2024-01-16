#pragma once

#include "structs.h"
#include "ray.h"

#define PI 3.1415

class camera {
public:
    __host__ __device__ camera(vec3_t lookfrom, vec3_t lookat, vec3_t vup, float vfov, float aspect);
    __host__ __device__ ray get_ray(float u, float v);

    vec3_t origin;
    vec3_t lower_left_corner;
    vec3_t horizontal;
    vec3_t vertical;
};