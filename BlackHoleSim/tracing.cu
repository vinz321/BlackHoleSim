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
    vec3_t col= r.march(ls, sphere({.3f,2,4}, 0.2,{0,0,0}), count);
    img(j, i) = col;
}

__global__ void render_shared(cv::cuda::PtrStepSz<vec3_t> img, cv::cuda::PtrStepSz<vec3_t> hdr, int max_x, int max_y, camera_t* cam_o, sphere_t* ls, int count, disk_t* disk_s) {
    __shared__ sphere_t spheres[5];
    __shared__ disk_t disk[1];
    __shared__ sphere_t blackhole[1];
    __shared__ camera_t cam[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < count; i++) {
                spheres[i] = ls[i];
           }

        blackhole[0] = sphere_t{ { 0,0,0 }, 0.2f, { 0,0,0 } , 0.0045f };
        disk[0] = *disk_s;

        cam[0] = camera_t{ cam_o->origin, cam_o->lower_left_corner, cam_o->horizontal, cam_o->vertical };
    }

    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //sphere s = sphere(*ls[0]);
    if ((i >= max_x) || (j >= max_y)) return;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);

    //ray r = ray(cam->origin, cam->lower_left_corner + (u * cam->horizontal) + (v * cam->vertical) - cam->origin, i, j, hdr);
    ray_t r1 = ray_t{ cam_o->origin, cam_o->lower_left_corner + (u * cam_o->horizontal) + (v * cam_o->vertical) - cam_o->origin , 0.02f, 256 };
    //vec3_t col = r.march(ls, blackhole, count, disk);
    vec3_t col = march(r1, hdr, ls, blackhole, count, disk);
    img(j, i) = col;
}

__global__ void instantiate_scene(sphere ** ls, int count) {
    for (int i = 0; i < count; i++) {
        ls[i] = new sphere(*(ls[i]));
    }
}

sphere** createScene(float angle) {
    int size = 2;

    sphere** scene = (sphere**)malloc(sizeof(sphere*)*size);
    sphere** scene_gpu;

    scene[0] = sphere(vec3_t{ 0, 2* sinf(angle)+2, 2*cosf(angle)+4 }, 0.3f, {0,1,0}).allocGPU();
    scene[1] = sphere(vec3_t{1,2,4}, 0.2f, {0, 1, 1}).allocGPU();

    cudaMalloc(&scene_gpu, sizeof(sphere*) * size);
    cudaMemcpy(scene_gpu, scene, sizeof(sphere*) * size, cudaMemcpyHostToDevice);

    instantiate_scene <<<1, 1 >>> (scene_gpu, 2);

    return scene_gpu;
}

sphere_t* createSceneStruct(float angle) {
    int size = 2;

    sphere_t* scene;
    cudaMallocHost(&scene, sizeof(sphere_t) * size + sizeof(disk_t));
    sphere_t* scene_gpu;
    cudaMalloc(&scene_gpu, sizeof(sphere_t) * size + sizeof(disk_t));


    scene[0] = sphere_t{ vec3_t{ -.7f * cosf(angle) , .7f * sinf(angle), 0 }, 0.1f, { 1,1,.8f }, 0 };
    scene[1] = sphere_t{vec3_t{ 0.9f,0,0 }, 0.05f, {.9f ,1, 1 }, 0};
    *(disk_t*)(scene + 2) = disk_t{ {0,0,0}, 0.25f, 0.6f, {1,1,1} ,{0,0,1} };

    cudaMemcpy(scene_gpu, scene, sizeof(sphere_t) * size+ sizeof(disk_t), cudaMemcpyHostToDevice);

    return scene_gpu;
}

void freeScene(sphere ** scene, int count) {
    cudaFree(scene);
}

void freeScene(sphere_t* scene) {
    cudaFree(scene);
}

cv::Mat3f renderScene(int img_w, int img_h, camera *cam, float &angle, Mat3f &hdr, sphere** scene) {
    cv::Mat3f img(img_h, img_w);
    cv::cuda::GpuMat gpu_img;
    cv::cuda::GpuMat gpu_hdr;

    //cv::resize(hdr, hdr, Size(img_w, img_h));

    gpu_img.upload(img);
    gpu_hdr.upload(hdr);

    dim3 grid_size(img_w/8,img_h/8);
    dim3 block_size(8,8);
    camera *cam_gpu;

    cudaMalloc(&cam_gpu, sizeof(camera));
    cudaMemcpy(cam_gpu, cam, sizeof(camera), cudaMemcpyHostToDevice);
    
    
    render <<<grid_size, block_size>>> (gpu_img, gpu_hdr, img_w, img_h, cam_gpu, scene, 2);

    cudaDeviceSynchronize();
    cudaFree(cam_gpu);
    //printf("%s \n", cudaGetErrorString(cudaGetLastError()));
    gpu_img.download(img);
    return img;
}


cv::Mat renderScene(int img_w, int img_h, camera_t* cam, float& angle, Mat3f& hdr, sphere_t* scene, disk_t* disk) {
    cv::Mat3f img(img_h, img_w);
    cv::cuda::GpuMat gpu_img;
    cv::cuda::GpuMat gpu_hdr;

    //cv::resize(hdr, hdr, Size(img_w, img_h));

    gpu_img.upload(img);
    gpu_hdr.upload(hdr);

    dim3 grid_size(img_w / 32, img_h / 32);
    dim3 block_size(32, 32);
    camera_t* cam_gpu;

    cudaMalloc(&cam_gpu, sizeof(camera_t));
    cudaMemcpy(cam_gpu, cam, sizeof(camera_t), cudaMemcpyHostToDevice);

    render_shared << < grid_size, block_size >> > (gpu_img, gpu_hdr, img_w, img_h, cam_gpu, scene, 2, (disk_t*)disk);
    //printf("%s \n", cudaGetErrorString(cudaGetLastError()));

    //cudaDeviceSynchronize();
    gpu_img.download(img);
    cudaFree(cam_gpu);
    //freeScene(scene, 2);
    return img;
}