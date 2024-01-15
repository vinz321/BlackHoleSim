#include "ray.h"
#include "sphere.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structs.h"
#include "camera.h"

__global__ void render(vec3** img, int max_x, int max_y, camera* cam);

__device__ vec3 px_color(ray& r);