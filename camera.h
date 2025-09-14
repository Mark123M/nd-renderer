#ifndef CAMERA_H
#define CAMERA_H

#include <sstream>
#include <vector>
#include <ctime>
#include <mutex>

#include "color.h"
#include "ray.h"
#include "util.h"
#include "light.h"
#include "transform.h"

static inline std::tm localtime_xp(std::time_t timer) {
    std::tm bt{};
#if defined(__unix__)
    localtime_r(&timer, &bt);
#elif defined(_MSC_VER)
    localtime_s(&bt, &timer);
#else
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    bt = *std::localtime(&timer);
#endif
    return bt;
}

// "YYYY-MM-DD__HH-MM-SS"
static inline std::string time_stamp(const std::string& fmt = "%F__%H-%M-%S") {
    auto bt = localtime_xp(std::time(0));
    char buf[64];
    return { buf, std::strftime(buf, sizeof(buf), fmt.c_str(), &bt) };
}

device_list<shape> scene;
__constant__ size_t d_scene_len; // static scenes for now

device_list<light> lights;
__constant__ size_t d_lights_len;

device_list<material> materials;
__constant__ size_t d_materials_len;

std::vector<uchar4> image_data;

namespace camera {

float aspect_ratio = 1.f;
uint image_width, image_height;
__constant__ uint d_image_width, d_image_height;
float viewport_width, viewport_height;
__constant__ float d_viewport_width, d_viewport_height;
bool render_normals = false;
__constant__ bool d_render_normals;

transform camera_transform{identity_affine, identity_affine};
__device__ transform d_camera_transform;
vec4 pixel_x;
__device__ vec4 d_pixel_x;
vec4 pixel_y;
__device__ vec4 d_pixel_y;
point4 pixel00_center;
__device__ point4 d_pixel00_center;

__device__ float scene_sdf_cuda(const point4& p, shape** shared_scene, light** shared_lights, shape** target_ptr) {
    float sdf = MAX_MARCH_DIST + 5.f;
    
    for (size_t i = 0; i < d_scene_len; i++) {
        shape* obj = shared_scene[i];
        float obj_sdf = obj->sdf(p);

        if (obj_sdf < sdf) {
            sdf = obj_sdf;

            if (target_ptr) {
                *target_ptr = obj; // assign object pointer
            }

            if (sdf <= TOL) {
                return sdf;
            }
        }
    }

    assert(sdf >= -TOL); // only reflections
    return sdf;
}

__device__ float scene_sdf_mod_cuda(const point4& p, shape** shared_scene, light** shared_lights, shape** target_ptr = nullptr) {
    //float s = 5.f;
    //point4 q(p.x - s * roundf(p.x / s), p.y - s * roundf(p.y / s), p.z - s * roundf(p.z / s), p.w);
    return scene_sdf_cuda(p, shared_scene, shared_lights, target_ptr);
}

__device__ vec4 get_normal_cuda(const point4& p, shape** shared_scene, light** shared_lights) {
    float sdf_diff_x = scene_sdf_mod_cuda(p + d_delta_x, shared_scene, shared_lights) - scene_sdf_mod_cuda(p - d_delta_x, shared_scene, shared_lights);
    float sdf_diff_y = scene_sdf_mod_cuda(p + d_delta_y, shared_scene, shared_lights) - scene_sdf_mod_cuda(p - d_delta_y, shared_scene, shared_lights);
    float sdf_diff_z = scene_sdf_mod_cuda(p + d_delta_z, shared_scene, shared_lights) - scene_sdf_mod_cuda(p - d_delta_z, shared_scene, shared_lights);
    float sdf_diff_w = scene_sdf_mod_cuda(p + d_delta_w, shared_scene, shared_lights) - scene_sdf_mod_cuda(p - d_delta_w, shared_scene, shared_lights);

    return vec4::normalize(vec4(sdf_diff_x, sdf_diff_y, sdf_diff_z, sdf_diff_w));
}

__device__ void ray_march_cuda(ray& r, shape** target_ptr, shape** shared_scene, light** shared_lights) {
    //float a = 0.5f * (r.dir.y + 1.f);
    //return (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
    for (uint i = 0; i < MAX_MARCH_STEPS; i++) {
        float dist = scene_sdf_mod_cuda(r.pos, shared_scene, shared_lights, target_ptr);

        if (dist <= TOL) {
            return;
        } else if (dist > MAX_MARCH_DIST) {
            *target_ptr = nullptr; // no object hit
            return;
        } else {
            r.march(dist);
        }
    }
}

__device__ color ray_color_cuda(ray& r, shape** shared_scene, light** shared_lights, material** shared_materials, uint64_t& pcg_state) {
    color col(1.f, 1.f, 1.f);

    for (uint k = 0; k < MAX_RAY_BOUNCES; k++) {
        hit_result res; // res is initialized with MAX_RAY_DIST
        // ray_march_cuda(r, &target, shared_scene, shared_lights);
        bool hit = false;
        
        // TODO: replace with BVH
        for (size_t i = 0; i < d_scene_len; i++) {
            hit = shared_scene[i]->intersect(r, res) || hit; // always evaluate intersect to find closer shapes
        }

        if (!hit) {
            float a = 0.5f * (r.dir.y + 1.f);
            return color(0.f, 0.f, 0.f); //* (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
        }

        const vec4& normal = res.normal;

        if (d_render_normals) {
            return 0.5f * color(normal.x + 1.f, normal.y + 1.f, normal.z + 1.f);
        }
        
        bsdf_sample bs;
        material* mat = shared_materials[res.target->mat_idx];
        if (!mat->sample_f(-r.dir, res.m, bs, pcg_state)) { // material is not reflective, return
            return col * bs.f;
        } else {
            col *= bs.f;
        }
        r = ray(res.p + EPSILON * bs.wi, bs.wi);
        
        /* color total_lighting(0.f, 0.f, 0.f);
        for (size_t i = 0; i < d_lights_len; i++) {
            light* lig = shared_lights[i];
            total_lighting += lig->Le_cuda(r.pos, normal, shared_scene, shared_lights);
            //printf("[GPU] Hit pos (%.3f, %.3f, %.3f, %.3f) | Total lighting (%.3f, %.3f, %.3f)\n",
            //r.pos.x, r.pos.y, r.pos.z, r.pos.w, total_lighting.r, total_lighting.g, total_lighting.b);
        } 

        return target->albedo * total_lighting; */
    }

    return color(0.f, 0.f, 0.f);
}

__global__ void render_kernel(uchar4* d_image_data, uint width, uint height, char* d_scene_data, size_t h_total_scene_bytes, char* d_lights_data, size_t h_total_lights_bytes, char* d_materials_data, size_t h_total_materials_bytes) {
    extern __shared__ char buffer[];

    // buffer layout for shared memory
    // total_scene_bytes d_scene_len * 8 total_lights_bytes d_lights_len * 8    
    // [all shape data | shape pointers | all light data | light pointers | all material data | material pointers]
    char* cur_shape_data = buffer;
    shape** shared_scene = (shape**)(cur_shape_data + h_total_scene_bytes);
    char* cur_light_data = (char*)(shared_scene + d_scene_len);
    light** shared_lights = (light**)(cur_light_data + h_total_lights_bytes);
    char* cur_material_data = (char*)(shared_lights + d_lights_len);
    material** shared_materials = (material**)(cur_material_data + h_total_materials_bytes);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        memcpy(cur_shape_data, d_scene_data, h_total_scene_bytes);
        memcpy(cur_light_data, d_lights_data, h_total_lights_bytes);
        memcpy(cur_material_data, d_materials_data, h_total_materials_bytes);
        
        for (size_t i = 0; i < d_scene_len; i++) {
            shared_scene[i] = (shape*)cur_shape_data;
            cur_shape_data += shared_scene[i]->size();
        }

        for (size_t i = 0; i < d_lights_len; i++) {
            shared_lights[i] = (light*)cur_light_data;
            cur_light_data += shared_lights[i]->size();
        }

        for(size_t i = 0; i < d_materials_len; i++) {
            shared_materials[i] = (material*)cur_material_data;
            cur_material_data += shared_materials[i]->size();
        }
    }

    __syncthreads();

    uint row = blockDim.x * blockIdx.x + threadIdx.x;
    uint col = blockDim.y * blockIdx.y + threadIdx.y;
    uint idx = row * width + col;

    if (row < height && col < width) {
        // need device versions of these functions/structs
        //point4 pixel_center = d_pixel00_center + (col * d_pixel_x) + (row * d_pixel_y);
        //vec4 direction = vec4::normalize(pixel_center - d_camera_transform.get_pos());
        //ray r(d_camera_transform.get_pos(), direction);

        uint64_t pcg_state = 0x4d595df4d0f33173;
        pcg32_init(idx, pcg_state);
        point4 camera_pos = d_camera_transform.get_pos();
        color c(0.f, 0.f, 0.f);

        // TODO: progressive renderer that draws k SPP every frame until total is reached (improve frame rates and interactions)
        for (uint s = 0; s < SAMPLES_PER_PIXEL; s++) {
            float row_offset = randf_pcg32(-0.5f, 0.5f, pcg_state);
            float col_offset = randf_pcg32(-0.5f, 0.5f, pcg_state);
            //vec4 offset{ randf_pcg32(-0.5f, 0.5f, pcg_state), randf_pcg32(-0.5f, 0.5f, pcg_state), 0.f, 0.f }; // Random point from center of unit square -0.5 <= x, y < 0.5
            point4 sample = d_pixel00_center + ((col + col_offset) * d_pixel_x) + ((row + row_offset) * d_pixel_y);
            vec4 direction = vec4::normalize(sample - camera_pos);
            ray r(camera_pos, direction);
            c += ray_color_cuda(r, shared_scene, shared_lights, shared_materials, pcg_state);
        }

        d_image_data[idx].x = static_cast<unsigned char>(fminf(1.f, c.r / SAMPLES_PER_PIXEL) * 255.999f);
        d_image_data[idx].y = static_cast<unsigned char>(fminf(1.f, c.g / SAMPLES_PER_PIXEL) * 255.999f);
        d_image_data[idx].z = static_cast<unsigned char>(fminf(1.f, c.b / SAMPLES_PER_PIXEL) * 255.999f);
        d_image_data[idx].w = 255;
    }
}

__global__ void render_stride_kernel(uchar4* d_image_data, uint width, uint num_pixels) {
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < num_pixels; 
         i += blockDim.x * gridDim.x) {
        uint row = i / width, col = i % width;
        point4 pixel_center = d_pixel00_center + (col * d_pixel_x) + (row * d_pixel_y);
        vec4 direction = vec4::normalize(pixel_center - d_camera_transform.get_pos());
        ray r(d_camera_transform.get_pos(), direction);
        
        /*color color = ray_color_cuda(r);
        d_image_data[i].x = static_cast<unsigned char>(fminf(1.f, color.r) * 255.999f);
        d_image_data[i].y = static_cast<unsigned char>(fminf(1.f, color.g) * 255.999f);
        d_image_data[i].z = static_cast<unsigned char>(fminf(1.f, color.b) * 255.999f);
        d_image_data[i].w = 255; */
    
    }
}

float scene_sdf(const point4& p, shape** target_ptr) {
    float sdf = MAX_MARCH_DIST + 5.f;
    
    for (size_t i = 0; i < scene.size(); i++) {
        shape* obj = scene[i];
        float obj_sdf = obj->sdf(p);

        if (obj_sdf < sdf) {
            sdf = obj_sdf;
            *target_ptr = obj; // assign object pointer
        }
    }

    assert(sdf >= -TOL); // only reflections
    return sdf;
}

float scene_sdf(const point4& p) {
    float sdf = MAX_MARCH_DIST + 5.f;

    for (size_t i = 0; i < scene.size(); i++) {
        shape* obj = scene[i];
        float obj_sdf = obj->sdf(p);

        if (obj_sdf < sdf) {
            sdf = obj_sdf;
        }
    }

    assert(sdf >= -TOL); // only reflections
    return sdf;
}

vec4 get_normal(const point4& p) {
    float sdf_diff_x = scene_sdf(p + delta_x) - scene_sdf(p - delta_x);
    float sdf_diff_y = scene_sdf(p + delta_y) - scene_sdf(p - delta_y);
    float sdf_diff_z = scene_sdf(p + delta_z) - scene_sdf(p - delta_z);
    float sdf_diff_w = scene_sdf(p + delta_w) - scene_sdf(p - delta_w);

    return vec4::normalize(vec4(sdf_diff_x, sdf_diff_y, sdf_diff_z, sdf_diff_w));
}

void ray_march(ray& r, shape** target_ptr) {
    //float a = 0.5f * (r.dir.y + 1.f);
    //return (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
    do {
        float dist = scene_sdf(r.pos, target_ptr);

        if (dist <= TOL) {
            return;
        } else if (dist > MAX_MARCH_DIST) {
            *target_ptr = nullptr; // no object hit
            return;
        } else {
            r.march(dist);
        }
    } while (true);
}

color ray_color(ray& r) {
    shape* target = nullptr;
    ray_march(r, &target);

    if (target == nullptr) {
        vec4 unit_direction = vec4::normalize(r.dir);
        float a = 0.5f * (unit_direction.y + 1.f);
        return (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
    }

    vec4 normal = get_normal(r.pos);

    if (render_normals) {
        return 0.5f * color(normal.x + 1.f, normal.y + 1.f, normal.z + 1.f);
    }

    color total_lighting(0.f, 0.f, 0.f);

    for (size_t i = 0; i < lights.len; i++) {
        light* lig = lights[i];
        total_lighting += lig->Le(r.pos, normal);
        //printf("[CPU] Hit pos (%.3f, %.3f, %.3f, %.3f) | Total lighting (%.3f, %.3f, %.3f)\n",
        //r.pos.x, r.pos.y, r.pos.z, r.pos.w, total_lighting.r, total_lighting.g, total_lighting.b);
    }

    //color final_col = target->albedo * total_lighting;
    //return final_col;
    return total_lighting;
}

void refresh_view() {
    vec4 viewport_x = viewport_width * camera_transform.get_vec_x();
    vec4 viewport_y = -viewport_height * camera_transform.get_vec_y();
    pixel_x = viewport_x / image_width;
    pixel_y = viewport_y / image_height;

    point4 viewport_top_left = camera_transform.get_pos() - camera_transform.get_vec_z() - viewport_x / 2.f - viewport_y / 2.f;
    pixel00_center = viewport_top_left + 0.5f * (pixel_x + pixel_y);
}

__device__ void refresh_view_cuda() {
    vec4 viewport_x = d_viewport_width * d_camera_transform.get_vec_x();
    vec4 viewport_y = -d_viewport_height * d_camera_transform.get_vec_y();
    d_pixel_x = viewport_x / d_image_width;
    d_pixel_y = viewport_y / d_image_height;

    point4 viewport_top_left = d_camera_transform.get_pos() - d_camera_transform.get_vec_z() - viewport_x / 2.f - viewport_y / 2.f;
    d_pixel00_center = viewport_top_left + 0.5f * (d_pixel_x + d_pixel_y);
}

__global__ void refresh_view_cuda_kernel() {
    refresh_view_cuda();
}

void initialize() {
    image_height = std::max(1.f, image_width / aspect_ratio);
    viewport_height = 2.f;
    viewport_width = viewport_height * ((float)image_width / image_height);
    camera_transform.set_pos(point4(0.f, 0.f, 1.f, 0.f));

    gpuErrchk(cudaMemcpyToSymbol(d_render_normals, &render_normals, sizeof(bool)));
    gpuErrchk(cudaMemcpyToSymbol(d_image_width, &image_width, sizeof(uint)));
    gpuErrchk(cudaMemcpyToSymbol(d_image_height, &image_height, sizeof(uint)));
    gpuErrchk(cudaMemcpyToSymbol(d_viewport_width, &viewport_width, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_viewport_height, &viewport_height, sizeof(float)));
    gpuErrchk(cudaMemcpyToSymbol(d_camera_transform, &camera_transform, sizeof(transform)));

    refresh_view();
    refresh_view_cuda_kernel<<<1,1>>>();

    gpuErrchk(cudaMemcpyToSymbol(d_delta_x, &delta_x, sizeof(vec4)));
    gpuErrchk(cudaMemcpyToSymbol(d_delta_y, &delta_y, sizeof(vec4)));
    gpuErrchk(cudaMemcpyToSymbol(d_delta_z, &delta_z, sizeof(vec4)));
    gpuErrchk(cudaMemcpyToSymbol(d_delta_w, &delta_w, sizeof(vec4)));
}

void render_rt(uint first_row, uint last_row) {
    // assume camera parameters are all initialized
    for (uint row = first_row; row <= last_row; row++) {
        //std::clog << "\rScanlines remaining: " << (image_height - j) << "     " << std::flush;
        for (uint col = 0; col < image_width; col++) {
            uint idx = row * image_width + col;
            point4 pixel_center = pixel00_center + (col * pixel_x) + (row * pixel_y);
            vec4 direction = vec4::normalize(pixel_center - camera_transform.get_pos());
            ray r(camera_transform.get_pos(), direction);
            
            color color = ray_color(r);
            image_data[idx].x = static_cast<unsigned char>(fminf(1.f, color.r) * 255.999f);
            image_data[idx].y = static_cast<unsigned char>(fminf(1.f, color.g) * 255.999f);
            image_data[idx].z = static_cast<unsigned char>(fminf(1.f, color.b) * 255.999f);
            image_data[idx].w = 255;
        }
    }
}

void render() {
    initialize();

    std::ofstream file{ "renders/render_" + time_stamp() + ".ppm", std::ios::app };

    clock_t t0 = clock();
    file << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (uint j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << "     " << std::flush;
        for (uint i = 0; i < image_width; i++) {
            point4 pixel_center = pixel00_center + (i * pixel_x) + (j * pixel_y);
            vec4 direction = vec4::normalize(pixel_center - camera_transform.get_pos());
            ray r(camera_transform.get_pos(), direction);

            color color = ray_color(r);
            write_color(file, color);
        }
    }

    int duration = (clock() - t0) / 1000;
    std::clog << "\rDone " << duration << "ms                                                    \n";

    std::stringstream ss;
    ss << duration / 1000;
    std::string duration_string = ss.str();
    std::ofstream metadata{ "renders/render_" + time_stamp() + "_" + duration_string + "s.metadata.txt", std::ios::app };
    metadata << "time elapsed: " << duration << " ms/" << duration / 1000.0 << " s/" << duration / 60000.0 << " m" << std::endl;
    metadata << "width: " << image_width << " height: " << image_height << "  " << std::endl;
    //metadata << "samples: " << samples_per_pixel << " max depth: " << max_depth << std::endl;

    file.close();
    metadata.close();
}

}

#endif