#ifndef WORLD_H
#define WORLD_H

#include <cuda_runtime.h>
#include "camera.h"
#include "direction_light.h"
#include "lambertian.h"
#include "specular.h"
#include "dielectric.h"

namespace world {

__global__ void translate_kernel(vec4 t, shape** scene_list) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < d_scene_len) {
        scene_list[idx]->translate(t);
    }
}

__global__ void move_camera_x_kernel(float amount) {
    camera::d_camera_transform.translate(amount * camera::d_camera_transform.get_vec_x());
    camera::refresh_view_cuda();
}

__global__ void move_camera_z_kernel(float amount) {
    camera::d_camera_transform.translate(amount * camera::d_camera_transform.get_vec_z());
    camera::refresh_view_cuda();
}

__global__ void move_camera_w_kernel(float amount) {
    camera::d_camera_transform.translate(amount * camera::d_camera_transform.get_vec_w());
    camera::refresh_view_cuda();
}

__global__ void rotate_camera_horizontal_kernel(float angle) {
    camera::d_camera_transform.rotate_xz(angle);
    camera::refresh_view_cuda();
}

__global__ void rotate_camera_vertical_kernel(float angle) {
    camera::d_camera_transform.rotate_yz(-angle);
    camera::refresh_view_cuda();
}

__global__ void rotate_xy_kernel(float angle, shape** scene_list) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        scene_list[idx]->rotate_xy(angle);
    }
}

__global__ void rotate_xz_kernel(float angle, shape** scene_list) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        scene_list[idx]->rotate_xz(angle);
    }
}

__global__ void rotate_xw_kernel(float angle, shape** scene_list) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        scene_list[idx]->rotate_xw(angle);
    }
}

__global__ void rotate_yz_kernel(float angle, shape** scene_list) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        scene_list[idx]->rotate_yz(angle);
    }
}

__global__ void rotate_yw_kernel(float angle, shape** scene_list) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        scene_list[idx]->rotate_yw(angle);
    }
}

__global__ void rotate_zw_kernel(float angle, shape** scene_list) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        scene_list[idx]->rotate_zw(angle);
    }
}
}

#endif