#ifndef WORLD_H
#define WORLD_H

#include <cuda_runtime.h>
#include "camera.h"
#include "direction_light.h"

enum shape_type {
    CYLINDER, HYPERSPHERE, HYPERCUBE
};

struct shape_params {
    shape_type type;
    transform basis;
    color albedo;
    
    point4 sphere_center;
    point4 cube_corner;
    point4 cylinder_a;
    point4 cylinder_b;
    bool cylinder_should_project;
    float radius;
};

enum light_type {
    DIRECTION, POINT, AREA
};

struct light_params {
    light_type type;
    vec4 dir;
	color col;
};

namespace world {

__global__ void construct(shape_params* d_scene_params_list, light_params* d_lights_params_list) {
    d_scene = new shape*[d_scene_len];
    
    for (size_t i = 0; i < d_scene_len; i++) {
        shape_params& params = d_scene_params_list[i];
        
        if (params.type == shape_type::CYLINDER) {
            cylinder* c = new cylinder;
            c->basis = params.basis;
            c->albedo = params.albedo;

            c->a = params.cylinder_a;
            c->b = params.cylinder_b;
            c->should_project = params.cylinder_should_project;
            c->radius = params.radius;
            d_scene[i] = c;
        } else if (params.type == shape_type::HYPERCUBE) {
            ncube* nc = new ncube;
            nc->basis = params.basis;
            nc->albedo = params.albedo;

            nc->corner = params.cube_corner;
            d_scene[i] = nc;
        } else if (params.type == shape_type::HYPERSPHERE) {
            nsphere* ns = new nsphere;
            ns->basis = params.basis;
            ns->albedo = params.albedo;

            ns->center = params.sphere_center;
            ns->radius = params.radius;
            d_scene[i] = ns;
        }
    }

    for (size_t i = 0; i < d_scene_len; i++) {
        d_scene[i]->print_gpu();
    }

    d_lights = new light*[d_lights_len];

    for (size_t i = 0; i < d_lights_len; i++) {
        light_params& params = d_lights_params_list[i];

        if (params.type == light_type::DIRECTION) {
            direction_light* dl = new direction_light;
            dl->col = params.col;
            dl->dir = params.dir;
            d_lights[i] = dl;
        }
    }

    for (size_t i = 0; i < d_lights_len; i++) {
        d_lights[i]->print_gpu();
    }
}

__global__ void destruct() {
    for (size_t i = 0; i < d_scene_len; i++) {
        delete d_scene[i];
    }

    delete [] d_scene;

    for (size_t i = 0; i < d_lights_len; i++) {
        delete d_lights[i];
    }

    delete [] d_lights;
}

__global__ void translation_kernel(vec4 t) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < d_scene_len) {
        d_scene[idx]->basis.translate(t);
    }
}

void initialize() {
    size_t scene_len = scene.size();
    gpuErrchk(cudaMemcpyToSymbol(d_scene_len, &scene_len, sizeof(size_t)));
    shape_params* scene_params_list = new shape_params[scene_len];
    shape_params* d_scene_params_list;
    size_t scene_size = scene_len * sizeof(shape_params);

    for (size_t i = 0; i < scene_len; i++) {
        cylinder* cylinder_ptr = dynamic_cast<cylinder*>(scene[i]);
        nsphere* nsphere_ptr = dynamic_cast<nsphere*>(scene[i]);
        ncube* ncube_ptr = dynamic_cast<ncube*>(scene[i]);

        shape_params& params = scene_params_list[i];
        params.basis = scene[i]->basis;
        params.albedo = scene[i]->albedo;

        if (cylinder_ptr) {
            params.type = shape_type::CYLINDER;
            params.cylinder_a = cylinder_ptr->a;
            params.cylinder_b = cylinder_ptr->b;
            params.cylinder_should_project = cylinder_ptr->should_project;
            params.radius = cylinder_ptr->radius;
        } else if (nsphere_ptr) {
            params.type = shape_type::HYPERSPHERE;
            params.sphere_center = nsphere_ptr->center;
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

    construct<<<1, 1>>>(d_scene_params_list, d_lights_params_list);
    delete [] scene_params_list;
    gpuErrchk(cudaFree(d_scene_params_list));
    delete [] lights_params_list;
    gpuErrchk(cudaFree(d_lights_params_list));
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