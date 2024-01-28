#pragma once

#include "tracing.h"
#include "sky_engine.h"
#include <iostream>

using namespace std;

__constant__ sphere_t* const_scene;

__global__ void render_shared(cv::cuda::PtrStepSz<vec3_t> img, cv::cuda::PtrStepSz<vec3_t> hdr, int max_x, int max_y, camera_t* cam_o, sphere_t* ls, int count, disk_t* disk_s) {
    //__shared__ sphere_t spheres[3];
    //__shared__ disk_t disk[1];
    //__shared__ camera_t cam[1];
    //if (threadIdx.x == 0 && threadIdx.y == 0) {
    //    for (int i = 0; i < count; i++) {
    //            spheres[i] = ls[i];
    //       }

    //    //blackhole[0] = sphere_t{ { 0,0,0 }, 0.2f, { 0,0,0 } , 0.0045f };
    //    
    //    disk[0] = *disk_s;

    //    cam[0] = camera_t{ cam_o->origin, cam_o->lower_left_corner, cam_o->horizontal, cam_o->vertical };
    //}

    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray_t r1 = ray_t{ cam_o->origin, cam_o->lower_left_corner + (u * cam_o->horizontal) + (v * cam_o->vertical) - cam_o->origin , 0.02f, 256 };
    vec3_t col = march(r1, hdr, ls, count, disk_s);
    img(j, i) = col;
}

sphere_t* createSceneStruct(float angle) {
    int size = 3;
    //sphere_t* scene_gpu;
    sphere_t* scene;
    cudaMallocHost(&scene, sizeof(sphere_t) * (size) + sizeof(disk_t));
    //cudaMalloc(&const_scene, sizeof(sphere_t) * (size) + sizeof(disk_t));

    scene[0] = sphere_t{ { 0,0,0 }, 0.3f, { 0,0,0 } , 0.0045f }; //blackhole
    scene[1] = sphere_t{ vec3_t{ -.7f * cosf(angle) , .7f * sinf(angle), 0 }, 0.1f, { 1,1,.8f }, 0};
    scene[2] = sphere_t{vec3_t{ 0.9f,0,0 }, 0.05f, {.9f ,1, 1 }, 0};
    *(disk_t*)(scene + 3) = disk_t{ {0,0,0}, 0.25f, 0.6f, {1,1,1} ,{0,0,1}};

    cudaMemcpyToSymbol(const_scene, scene, sizeof(sphere_t) * (size)+sizeof(disk_t), cudaMemcpyHostToDevice);

    return const_scene;
}

cv::Mat renderScene(int img_w, int img_h, camera_t* cam, float& angle, Mat3f& hdr, sphere_t* scene, disk_t* disk) {
    cv::Mat3f img(img_h, img_w);
    cv::cuda::GpuMat gpu_img;
    cv::cuda::GpuMat gpu_hdr;

    gpu_img.upload(img);
    gpu_hdr.upload(hdr);

    dim3 grid_size(img_w / 32, img_h / 32);
    dim3 block_size(32, 32);
    camera_t* cam_gpu;

    cudaMalloc(&cam_gpu, sizeof(camera_t));
    cudaMemcpy(cam_gpu, cam, sizeof(camera_t), cudaMemcpyHostToDevice);

    render_shared <<< grid_size, block_size >>> (gpu_img, gpu_hdr, img_w, img_h, cam_gpu, scene, 3, (disk_t*)disk);
    //printf("%s \n", cudaGetErrorString(cudaGetLastError()));

    cudaDeviceSynchronize();
    gpu_img.download(img);
    cudaFree(cam_gpu);
    //freeScene(scene, 2);
    return img;
}