
#include "tracing.h"

__device__ vec3_t color(ray r) {
    if (r.hit_sphere(sphere(vec3_t{ 0.0f, 0.0f, 0.0f }, 0.5)))
        return vec3_t{ 1.0f, 0.0f, 0.0f };
    vec3_t unit_direction = norm(r.get_dir());
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * vec3_t{1.0, 1.0, 1.0} + t * vec3_t{ 0.5, 0.7, 1.0 };    
}

__global__ void render (cv::cuda::PtrStepSz<vec3_t> img, int max_x, int max_y, camera *cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    //int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r = ray(cam->origin, cam->lower_left_corner + (u * cam->horizontal) + (v * cam->vertical) - cam->origin);
    //vec3 t = cam.horizontal;
    //ray r = ray(vec3_t{ 0,0,0 }, vec3_t{ 0,0,1 });
    img(j,i) = color(r);
    //img(j, i) = vec3_t{ 0.5f,0.7f,1.0f };
}

cv::Mat3f renderScene() {
    cv::Mat3f img(512, 512);
    img.setTo(cv::Vec3f(0,0,0));

    cv::cuda::GpuMat gpu_img;

    gpu_img.upload(img);

    dim3 grid_size(16,16);
    dim3 block_size(32,32);
    camera cam(vec3_t{ 0,0,-2 }, vec3_t{ 0,0,1 }, vec3_t{ 0,1,0 }, 60, 1);

    camera *cam_gpu;
    cudaMalloc(&cam_gpu, sizeof(camera));
    cudaMemcpy(cam_gpu, &cam, sizeof(camera), cudaMemcpyHostToDevice);
    render <<<grid_size, block_size >>> (gpu_img, 512, 512, cam_gpu);
    printf("%s", cudaGetErrorString(cudaGetLastError()));


    cudaDeviceSynchronize();
    gpu_img.download(img);

    return img;
}

