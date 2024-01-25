#include "camera.h"

__host__ __device__ camera::camera(vec3_t lookfrom, vec3_t lookat, vec3_t vup, float vfov, float aspect) {
    // vfov is top to bottom in degrees
    vec3_t u, v, w;
    float theta = vfov * PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = norm(lookfrom - lookat);
    u = norm(cross(vup, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width * u - half_height * v - w;
    horizontal = 2 * half_width * u;
    vertical = 2 * half_height * v;
}