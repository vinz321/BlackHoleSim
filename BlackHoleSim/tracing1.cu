#include "tracing.h"



__device__ vec3 px_color(ray& r) {
    if (r.hit_sphere(sphere(vec3{ 0.0f, 0.0f, -1.0f }, 0.5)))
        return vec3{ 1.0f, 0.0f, 0.0f };
    vec3 unit_direction = norm(r.get_dir());
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * vec3 { 1.0, 1.0, 1.0 } + t * vec3{ 0.5, 0.7, 1.0 };
}

__global__ void render(vec3* img, int max_x, int max_y, camera* cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    cam = new camera(vec3{ 0.0f, 0.0f, -1.0f },
        vec3{ 0.0f, 0.0f, 1.0f },
        vec3{ 0.0f, 1.0f, 0.0f },
        60,
        max_x / max_y);

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r = cam->get_ray(u, v);
    img[pixel_index] = px_color(r);
}

