#pragma once

#include "structs.h"
#include "ray.h"

__host__ __device__ camera_t make_cam(vec3_t lookfrom, vec3_t lookat, vec3_t vup, float vfov, float aspect);__host__ __device__ camera_t make_cam(vec3_t lookfrom, vec3_t lookat, vec3_t vup, float vfov, float aspect);
//__host__ __device__ ray get_ray(camera_t c,float u, float v);//__host__ __device__ ray get_ray(camera_t c,float u, float v);//__host__ __device__ ray get_ray(camera_t c,float u, float v);//__host__ __device__ ray get_ray(camera_t c,float u, float v);//__host__ __device__ ray get_ray(camera_t c,float u, float v);//__host__ __device__ ray get_ray(camera_t c,float u, float v);