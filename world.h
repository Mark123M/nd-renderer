#ifndef WORLD_H
#define WORLD_H

#include <cuda_runtime.h>
#include "camera.h"
#include "direction_light.h"
#include "lambertian.h"
#include "specular.h"
#include "dielectric.h"

struct light_params {
    light_type type;
    vec4 dir;
	color col;
};

struct material_params {
    material_type type;
    color albedo;
    float eta;
};

namespace world {

__global__ void construct(light_params* d_lights_params_list, material_params* d_materials_params_list, size_t h_total_scene_bytes, size_t h_total_lights_bytes, size_t h_total_materials_bytes) {
    d_lights_data = new char[h_total_lights_bytes];
    char* cur_light_data = d_lights_data;
    d_lights = new light*[d_lights_len];

    for (size_t i = 0; i < d_lights_len; i++) {
        light_params& params = d_lights_params_list[i];

        if (params.type == light_type::DIRECTION) {
            direction_light* dl = new(cur_light_data) direction_light;
            cur_light_data += dl->size();
            dl->col = params.col;
            dl->dir = params.dir;
            d_lights[i] = dl;
        }
    }

    for (size_t i = 0; i < d_lights_len; i++) {
        d_lights[i]->print_gpu();
    }

    d_materials_data = new char[h_total_materials_bytes];
    char* cur_material_data = d_materials_data;
    d_materials = new material*[d_materials_len];

    for (size_t i = 0; i < d_materials_len; i++) {
        material_params& params = d_materials_params_list[i];

        if (params.type == material_type::LAMBERTIAN) {
            lambertian* l = new(cur_material_data) lambertian;
            cur_material_data += l->size();
            l->albedo = params.albedo;
            d_materials[i] = l;
        } else if (params.type == material_type::SPECULAR) {
            specular* s = new(cur_material_data) specular;
            cur_material_data += s->size();
            s->albedo = params.albedo;
            d_materials[i] = s;
        } else if (params.type == material_type::DIELECTRIC) {
            dielectric* d = new(cur_material_data) dielectric;
            cur_material_data += d->size();
            d->eta = params.eta;
            d_materials[i] = d;
        }
    }

    for (size_t i = 0; i < d_materials_len; i++) {
        d_materials[i]->print_gpu();
    }
}

__global__ void destruct() {
    delete [] d_lights_data;
    delete [] d_lights;

    delete [] d_materials_data;
    delete [] d_materials;
}

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

void initialize() {
    size_t scene_len = scene.size();
    gpuErrchk(cudaMemcpyToSymbol(d_scene_len, &scene_len, sizeof(size_t)));

    size_t lights_len = lights.size();
    gpuErrchk(cudaMemcpyToSymbol(d_lights_len, &lights_len, sizeof(size_t)));
    light_params* lights_params_list = new light_params[lights_len];
    light_params* d_lights_params_list;
    size_t lights_size = lights_len * sizeof(light_params);

    for (size_t i = 0; i < lights_len; i++) {
        direction_light* direction_ptr = dynamic_cast<direction_light*>(lights[i]);

        light_params& params = lights_params_list[i];
        params.col = lights[i]->col;

        if (direction_ptr) {
            params.type = light_type::DIRECTION;
            params.dir = direction_ptr->dir;
        }
    }

    gpuErrchk(cudaMalloc(&d_lights_params_list, lights_size));
    gpuErrchk(cudaMemcpy(d_lights_params_list, lights_params_list, lights_size, cudaMemcpyHostToDevice));

    size_t materials_len = materials.size();
    gpuErrchk(cudaMemcpyToSymbol(d_materials_len, &materials_len, sizeof(size_t)));
    material_params* materials_params_list = new material_params[materials_len];
    material_params* d_materials_params_list;
    size_t materials_size = materials_len * sizeof(material_params);

    for (size_t i = 0; i < materials_len; i++) {
        lambertian* lambertian_ptr = dynamic_cast<lambertian*>(materials[i]);
        specular* specular_ptr = dynamic_cast<specular*>(materials[i]);
        dielectric* dielectric_ptr = dynamic_cast<dielectric*>(materials[i]);

        material_params& params = materials_params_list[i];
        
        if (lambertian_ptr) {
            params.type = material_type::LAMBERTIAN;
            params.albedo = lambertian_ptr->albedo;
        } else if (specular_ptr) {
            params.type = material_type::SPECULAR;
            params.albedo = specular_ptr->albedo;
        } else if (dielectric_ptr) {
            params.type = material_type::DIELECTRIC;
            params.eta = dielectric_ptr->eta;
        }
    }

    gpuErrchk(cudaMalloc(&d_materials_params_list, materials_size));
    gpuErrchk(cudaMemcpy(d_materials_params_list, materials_params_list, materials_size, cudaMemcpyHostToDevice));

    construct<<<1, 1>>>(d_lights_params_list, d_materials_params_list, scene.data_size, total_lights_bytes, total_materials_bytes);
    
    delete [] lights_params_list;
    gpuErrchk(cudaFree(d_lights_params_list));
    delete [] materials_params_list;
    gpuErrchk(cudaFree(d_materials_params_list));
    //light_params* 

   /* std::vector<shape*> scene;
__device__ shape** d_scene;
__constant__ size_t d_scene_len;
std::vector<light*> lights;
__device__ light** d_lights;
__constant__ size_t d_lights_len;
std::vector<unsigned char> image_data;*/
}
}

#endif