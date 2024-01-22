#pragma once

#include "tracing.h"
#include "sky_engine.h"
#include <iostream>

using namespace std;

__device__ vec3_t color(ray r) {
    if (r.hit_sphere(sphere(vec3_t{ 0.0f, 0.0f, 0.0f }, 0.5)))
        return vec3_t{ 0.0f, 0.0f, 0.0f };
    //else return HDRI color correspondent to this ray
    vec3_t unit_direction = norm(r.get_dir());
    
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * vec3_t{1.0, 1.0, 1.0} + t * vec3_t{ 0.5, 0.7, 1.0 };
}

__global__ void render (cv::cuda::PtrStepSz<vec3_t> img, cv::cuda::PtrStepSz<vec3_t> hdr, int max_x, int max_y, camera *cam, sphere **ls, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    //sphere s = sphere(*ls[0]);
    if ((i >= max_x) || (j >= max_y)) return;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r = ray(cam->origin, cam->lower_left_corner + (u * cam->horizontal) + (v * cam->vertical) - cam->origin, i, j, hdr);
    vec3_t col= r.march(ls, sphere({.3f,0,0}, 0.2,{0,0,0}), count);
    img(j, i) = col;
}

__global__ void instantiate_scene(sphere ** ls, int count) {
    for (int i = 0; i < count; i++) {
        ls[i] = new sphere(*(ls[i]));
    }
}

sphere** createScene() {
    int size = 2;

    sphere** scene = (sphere**)malloc(sizeof(sphere*)*size);
    sphere** scene_gpu;

    scene[0] = sphere(vec3_t{0,0,2}, 0.3f, {0,1,0}).allocGPU();
    scene[1] = sphere(vec3_t{1,0,0}, 0.2f, {0, 1, 1}).allocGPU();

    cudaMalloc(&scene_gpu, sizeof(sphere*) * size);
    cudaMemcpy(scene_gpu, scene, sizeof(sphere*) * size, cudaMemcpyHostToDevice);

    instantiate_scene << <1, 1 >> > (scene_gpu, 2);

    return scene_gpu;
}

void freeScene(sphere ** scene, int count) {
    /*for (int i = 0; i < count; i++) {
        cudaFree(scene[i]);
        printf("%s \n", cudaGetErrorString(cudaGetLastError()));
    }*/
    cudaFree(scene);
    //printf("%s \n", cudaGetErrorString(cudaGetLastError()));
}
cv::Mat3f renderScene(int img_w, int img_h, camera *cam) {
    cv::Mat3f img(img_h, img_w);
    cv::Mat3f hdr = read_exr();
    cv::Mat3f hdr_rs;
    cv::cuda::GpuMat gpu_img;
    cv::cuda::GpuMat gpu_hdr;

    cv::resize(hdr, hdr_rs, Size(img_w, img_h));
    namedWindow("HDR", WINDOW_NORMAL);
    imshow("HDR", hdr);

    gpu_img.upload(img);
    gpu_hdr.upload(hdr_rs);

    gpu_hdr.download(hdr_rs);
    namedWindow("gpuHDR", WINDOW_NORMAL);
    imshow("gpuHDR", hdr_rs);

    dim3 grid_size(img_w/32,img_h/32);
    dim3 block_size(32,32);
    camera *cam_gpu;

    cudaMalloc(&cam_gpu, sizeof(camera));
    cudaMemcpy(cam_gpu, cam, sizeof(camera), cudaMemcpyHostToDevice);
    //printf("%s \n", cudaGetErrorString(cudaGetLastError()));
    sphere** scene = createScene();
    //cudaMalloc(&scene, sizeof(object) * 2);
    
    render <<<grid_size, block_size>>> (gpu_img, gpu_hdr, img_w, img_h, cam_gpu, scene, 2);

    cudaDeviceSynchronize();
    cudaFree(cam_gpu);
    //printf("%s \n", cudaGetErrorString(cudaGetLastError()));
    gpu_img.download(img);
    freeScene(scene, 2);
    return img;
}

