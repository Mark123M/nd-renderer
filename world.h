#ifndef WORLD_H
#define WORLD_H

#include <cuda_runtime.h>
#include "camera.h"
#include "direction_light.h"
#include "lambertian.h"
#include "specular.h"

struct shape_params {
    shape_type type;
    transform basis;
    color albedo;
    size_t mat_idx;
    
    point4 cube_corner;
    point4 cylinder_start0;
    point4 cylinder_end0;
    float radius;
};

struct light_params {
    light_type type;
    vec4 dir;
	color col;
};

struct material_params {
    material_type type;
    color albedo;
};

namespace world {

__global__ void construct(shape_params* d_scene_params_list, light_params* d_lights_params_list, material_params* d_materials_params_list, size_t h_total_scene_bytes, size_t h_total_lights_bytes, size_t h_total_materials_bytes) {
    d_scene_data = new char[h_total_scene_bytes];
    char* cur_shape_data = d_scene_data;
    d_scene = new shape*[d_scene_len];
    
    for (size_t i = 0; i < d_scene_len; i++) {
        shape_params& params = d_scene_params_list[i];
        
        if (params.type == shape_type::CYLINDER) {
            //cylinder* c = new cylinder;
            cylinder* c = new(cur_shape_data) cylinder;
            cur_shape_data += c->size();
            c->basis = params.basis;
            c->albedo = params.albedo;
            c->mat_idx = params.mat_idx;

            c->start0 = params.cylinder_start0;
            c->end0 = params.cylinder_end0;
            c->radius = params.radius;
            d_scene[i] = c;
        } else if (params.type == shape_type::PROJECTED_CYLINDER) {
            projected_cylinder* pc = new(cur_shape_data) projected_cylinder;
            cur_shape_data += pc->size();
            pc->basis = params.basis;
            pc->albedo = params.albedo;
            pc->mat_idx = params.mat_idx;

            pc->start0 = params.cylinder_start0;
            pc->end0 = params.cylinder_end0;
            pc->radius = params.radius;
            d_scene[i] = pc;
        } else if (params.type == shape_type::HYPERCUBE) {
            ncube* nc = new(cur_shape_data) ncube;
            cur_shape_data += nc->size();
            nc->basis = params.basis;
            nc->albedo = params.albedo;
            nc->mat_idx = params.mat_idx;

            nc->corner = params.cube_corner;
            d_scene[i] = nc;
        } else if (params.type == shape_type::HYPERSPHERE) {
            nsphere* ns = new(cur_shape_data) nsphere;
            cur_shape_data += ns->size();
            ns->basis = params.basis;
            ns->albedo = params.albedo;
            ns->mat_idx = params.mat_idx;

            ns->radius = params.radius;
            d_scene[i] = ns;
        }
    }

    for (size_t i = 0; i < d_scene_len; i++) {
        d_scene[i]->print_gpu();
    }

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
        }
    }

    for (size_t i = 0; i < d_materials_len; i++) {
        d_materials[i]->print_gpu();
    }
}

__global__ void destruct() {
    delete [] d_scene_data;
    delete [] d_scene;

    delete [] d_lights_data;
    delete [] d_lights;

    delete [] d_materials_data;
    delete [] d_materials;
}

__global__ void translate_kernel(vec4 t) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < d_scene_len) {
        d_scene[idx]->translate(t);
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

__global__ void rotate_xy_kernel(float angle) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        d_scene[idx]->rotate_xy(angle);
    }
}

__global__ void rotate_xz_kernel(float angle) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        d_scene[idx]->rotate_xz(angle);
    }
}

__global__ void rotate_xw_kernel(float angle) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        d_scene[idx]->rotate_xw(angle);
    }
}

__global__ void rotate_yz_kernel(float angle) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        d_scene[idx]->rotate_yz(angle);
    }
}

__global__ void rotate_yw_kernel(float angle) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        d_scene[idx]->rotate_yw(angle);
    }
}

__global__ void rotate_zw_kernel(float angle) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < d_scene_len) {
        d_scene[idx]->rotate_zw(angle);
    }
}

void initialize() {
    size_t scene_len = scene.size();
    gpuErrchk(cudaMemcpyToSymbol(d_scene_len, &scene_len, sizeof(size_t)));
    shape_params* scene_params_list = new shape_params[scene_len];
    shape_params* d_scene_params_list;
    size_t scene_size = scene_len * sizeof(shape_params);

    for (size_t i = 0; i < scene_len; i++) {
        shape* shape_ptr = scene[i];
        cylinder* cylinder_ptr = dynamic_cast<cylinder*>(shape_ptr);
        projected_cylinder* projected_cylinder_ptr = dynamic_cast<projected_cylinder*>(shape_ptr);
        nsphere* nsphere_ptr = dynamic_cast<nsphere*>(shape_ptr);
        ncube* ncube_ptr = dynamic_cast<ncube*>(shape_ptr);

        shape_params& params = scene_params_list[i];
        params.basis = shape_ptr->basis;
        params.albedo = shape_ptr->albedo;
        params.mat_idx = shape_ptr->mat_idx;

        if (cylinder_ptr) {
            params.type = shape_type::CYLINDER;
            params.cylinder_start0 = cylinder_ptr->start0;
            params.cylinder_end0 = cylinder_ptr->end0;
            params.radius = cylinder_ptr->radius;
        } else if (projected_cylinder_ptr) {
            params.type = shape_type::PROJECTED_CYLINDER;
            params.cylinder_start0 = projected_cylinder_ptr->start0;
            params.cylinder_end0 = projected_cylinder_ptr->end0;
            params.radius = projected_cylinder_ptr->radius;
        } else if (nsphere_ptr) {
            params.type = shape_type::HYPERSPHERE;
            params.radius = nsphere_ptr->radius;
        } else if (ncube_ptr) {
            params.type = shape_type::HYPERCUBE;
            params.cube_corner = ncube_ptr->corner;
        }
    }

    gpuErrchk(cudaMalloc(&d_scene_params_list, scene_size));
    gpuErrchk(cudaMemcpy(d_scene_params_list, scene_params_list, scene_size, cudaMemcpyHostToDevice));

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

        material_params& params = materials_params_list[i];
        
        if (lambertian_ptr) {
            params.type = material_type::LAMBERTIAN;
            params.albedo = lambertian_ptr->albedo;
        } else if (specular_ptr) {
            params.type = material_type::SPECULAR;
            params.albedo = specular_ptr->albedo;
        }
    }

    gpuErrchk(cudaMalloc(&d_materials_params_list, materials_size));
    gpuErrchk(cudaMemcpy(d_materials_params_list, materials_params_list, materials_size, cudaMemcpyHostToDevice));

    construct<<<1, 1>>>(d_scene_params_list, d_lights_params_list, d_materials_params_list, total_scene_bytes, total_lights_bytes, total_materials_bytes);
    
    delete [] scene_params_list;
    gpuErrchk(cudaFree(d_scene_params_list));
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