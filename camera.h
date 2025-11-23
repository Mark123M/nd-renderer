#ifndef CAMERA_H
#define CAMERA_H

#include <sstream>
#include <vector>
#include <ctime>
#include <mutex>
#include <math.h>

#include "color.h"
#include "ray.h"
#include "util.h"
#include "light.h"
#include "transform.h"
#include "bvh.h"

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

__device__ size_t d_scene_len; // static scenes for now
__device__ size_t d_lights_len;
__device__ size_t d_materials_len;

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

__device__ bool intersect_world(ray& r, hit_result& res, shape** shared_scene) {
    // ray_march_cuda(r, &target, shared_scene, shared_lights);
    bool hit = false;
    
    // TODO: replace with BVH
    for (size_t i = 0; i < d_scene_len; i++) {
        hit = shared_scene[i]->intersect(r, res) || hit; // always evaluate intersect to find closer shapes
    }

    return hit;
}

struct area_sample {
    float G; // geometry term
    color f; // bsdf reflectance
    hit_result res_next;
};

// computes f and G for a 
__device__ bool sample_area(
    area_sample& as,
    const shape* s,
    const material* mat_hit,
    const hit_result& res,
    linear_bvh_node* shared_linear_nodes,
    shape** shared_scene,
    uint64_t& pcg_state
) {
    shape_sample ss;
    s->sample(ss, res, pcg_state);

    vec4 oriented_normal = vec4::dot(res.wo, res.normal) < 0.f ? -res.normal : res.normal;
    point4 p_offset = res.p + EPSILON * oriented_normal;
    vec4 v = ss.p - p_offset; // vector to shape
    
    float dist = vec4::length(v);
    vec4 wi = v / dist;
    color f = mat_hit->f(res.wo, wi, res.m);
    ray r_next = ray(p_offset, wi);

    // visibility term
    hit_result res_next;
    if (!intersect_bvh(r_next, res_next, shared_linear_nodes, shared_scene) || !point4::approx_points_equals(res_next.p, ss.p)) {
        return false;
    }

    float cos_wi = vec4::dot(wi, res.normal); 
    float cos_wo_next = vec4::dot(-wi, res_next.normal);
    float G = (fabsf(cos_wi) * fabsf(cos_wo_next)) / (dist * dist * dist);

    as.G = G;
    as.f = f;
    as.res_next = res_next;
    return true;
}

__device__ color ray_color_cuda(
    ray& r,
    linear_bvh_node* shared_linear_nodes,
    shape** shared_scene,
    light** shared_lights,
    material** shared_materials,
    bool sample_lights,
    uint64_t& pcg_state
) {
    color L(0.f, 0.f, 0.f);
    //color col(1.f, 1.f, 1.f);
    color beta(1.f, 1.f, 1.f);
    bool specular_bounce = true;

    float total_surface_volume = 0.f;
    float total_emitter_surface_volume = 0.f;

    for (size_t i = 0; i < d_scene_len; i++) {
        float sv = shared_scene[i]->surface_volume();
        material* mat = shared_materials[shared_scene[i]->mat_idx];
        total_surface_volume += sv;

        if (mat->is_emissive()) {
            if (approx_equals(sv, 0.f)) { // can't sample emitters when sv function hasn't been implemented
                return color(0.f, 0.f, 0.f);
            }
            total_emitter_surface_volume += sv;
        }
    }


    for (uint k = 0; k < MAX_RAY_BOUNCES; k++) {
        hit_result res; // res is initialized with MAX_RAY_DIST

        if (!intersect_bvh(r, res, shared_linear_nodes, shared_scene)) {
            // float a = 0.5f * (r.dir.y + 1.f);
            // return color(0.5f, 0.5f, 0.5f); // constant environment map
            // return color(0.f, 0.f, 0.f); //* (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
            break;
        }

        material* mat = shared_materials[res.target->mat_idx];

        if (mat->is_emissive()) {
            if (!sample_lights || specular_bounce) {
                L += beta * mat->L();
            }
        }

        if (sample_lights && !mat->is_specular()) {
            shape* e = nullptr;
            material* light_mat = nullptr;
            float u = randf_pcg32(0.f, 1.f, pcg_state);

            for (size_t i = 0; i < d_scene_len; i++) {
                material* mat = shared_materials[shared_scene[i]->mat_idx];

                if (mat->is_emissive()) {
                    u -= shared_scene[i]->surface_volume() / total_emitter_surface_volume;

                    if (u <= 0) {
                        e = shared_scene[i];
                        light_mat = mat;
                        break;
                    }
                }
            }

            area_sample as_emitter;

            if (e != nullptr && sample_area(as_emitter, e, mat, res, shared_linear_nodes, shared_scene, pcg_state)) {
                // color L_throughput = ((as_emitter.G * as_emitter.f) / (1.f / total_emitter_surface_volume)) * beta;
               // vec4 li_wi = -as_emitter.res_next.wo;
               // color li_f = as_emitter.f * fabsf(vec4::dot(li_wi, res.normal));
                //L += (li_f * beta * light_mat->L()) / (1.f / total_emitter_surface_volume);
                //L += light_mat->L() * L_throughput;

                float pdf_A = 1.f / total_emitter_surface_volume;
                L += (as_emitter.f * beta * light_mat->L() * as_emitter.G) / pdf_A;
            }
        }
        
        bsdf_sample bs;

        if (!mat->sample_f(-r.dir, res.m, bs, pcg_state)) {
            break;
        }

        beta *= (bs.f * fabsf(vec4::dot(bs.wi, res.normal))) / bs.pdf;
        specular_bounce = mat->is_specular();
        r = ray(res.p + EPSILON * bs.wi, bs.wi);
        
        float beta_max = fmaxf(beta.r, fmaxf(beta.g, beta.b));
        if (beta_max <= 1.f && k >= 3) {
            float q = fmaxf(0.f, 1.f - beta_max);
            if (randf_pcg32(0.f, 1.f, pcg_state) < q) {
                break;
            }
            beta /= 1.f - q;
        }
    }

    return L;
}

__device__ color ray_color_area_cuda(
    ray& r,
    linear_bvh_node* shared_linear_nodes,
    shape** shared_scene,
    light** shared_lights,
    material** shared_materials,
    bool sample_lights,
    uint64_t& pcg_state
) {
    float total_surface_volume = 0.f;
    float total_emitter_surface_volume = 0.f;

    for (size_t i = 0; i < d_scene_len; i++) {
        float sv = shared_scene[i]->surface_volume();
        material* mat = shared_materials[shared_scene[i]->mat_idx];
        total_surface_volume += sv;

        if (mat->is_emissive()) {
            total_emitter_surface_volume += sv;
        }

        if (approx_equals(sv, 0.f)) { // can't area sample when sv function hasn't been implemented
            return color(0.f, 0.f, 0.f);
        }
    }

    color L(0.f, 0.f, 0.f);
    color beta(1.f, 1.f, 1.f);
    hit_result res;

    if (!intersect_bvh(r, res, shared_linear_nodes, shared_scene)) { // can't area sample when the world is unbounded
        return color(0.f, 0.f, 0.f);
    }

    uint k = 0;
    bool specular_bounce = true;
    material* mat_hit = shared_materials[res.target->mat_idx];

    if (sample_lights && mat_hit->is_emissive()) {
        L = mat_hit->L();
        // return mat_hit->L();
    }

    for (uint k = 0; k < MAX_RAY_BOUNCES; k++) {
        mat_hit = shared_materials[res.target->mat_idx];
        specular_bounce = mat_hit->is_specular();

        if (mat_hit->is_emissive()) {
            if (!sample_lights || specular_bounce) { // avoid NEE double count
                L += beta * mat_hit->L();
            }

            // break;
        }

        if (sample_lights) {
            shape* e = nullptr;
            material* light_mat = nullptr;
            float u = randf_pcg32(0.f, 1.f, pcg_state);

            for (size_t i = 0; i < d_scene_len; i++) {
                material* mat = shared_materials[shared_scene[i]->mat_idx];

                if (mat->is_emissive()) {
                    u -= shared_scene[i]->surface_volume() / total_emitter_surface_volume;

                    if (u <= 0) {
                        e = shared_scene[i];
                        light_mat = mat;
                        break;
                    }
                }
            }

            area_sample as_emitter;

            if (e != nullptr && sample_area(as_emitter, e, mat_hit, res, shared_linear_nodes, shared_scene, pcg_state)) {
                color L_throughput = ((as_emitter.G * as_emitter.f) / (1.f / total_emitter_surface_volume)) * beta;
                L += light_mat->L() * L_throughput;
            }
        }

        // sample point in scene
        shape* s = nullptr;
        float u = randf_pcg32(0.f, 1.f, pcg_state);

        for (size_t i = 0; i < d_scene_len; i++) {
            u -= shared_scene[i]->surface_volume() / total_surface_volume;

            if (u <= 0) {
                s = shared_scene[i];
                break;
            }
        }

        area_sample as;

        if (s == nullptr || !sample_area(as, s, mat_hit, res, shared_linear_nodes, shared_scene, pcg_state)) {
            break;
        }

        beta *= (as.G * as.f) / (1.f / total_surface_volume);
        res = as.res_next;

        float beta_max = fmaxf(beta.r, fmaxf(beta.g, beta.b));
        if (beta_max <= 1.f && k >= 2) {
            float q = fmaxf(0.f, 1.f - beta_max);
            if (randf_pcg32(0.f, 1.f, pcg_state) < q) {
                break;
            }
            beta /= 1.f - q;
        }
    }

    return L;
}

__global__ void render_kernel_global(
    int num_samples,
    int sample_mode,
    bool sample_lights,
    uint width,
    uint height,
    linear_bvh_node* d_linear_nodes,
    shape** d_scene,
    light** d_lights,
    material** d_materials,
    double3* d_color_buffer,
    uchar4* d_image_data,
    uint64_t* d_pcg_states
) {
    uint row = blockDim.x * blockIdx.x + threadIdx.x;
    uint col = blockDim.y * blockIdx.y + threadIdx.y;
    uint idx = row * width + col;

    if (row < height && col < width) {
        point4 camera_pos = d_camera_transform.get_pos();
        float row_offset = randf_pcg32(-0.5f, 0.5f, d_pcg_states[idx]);
        float col_offset = randf_pcg32(-0.5f, 0.5f, d_pcg_states[idx]);
        
        point4 sample = d_pixel00_center + ((col + col_offset) * d_pixel_x) + ((row + row_offset) * d_pixel_y);
        vec4 direction = vec4::normalize(sample - camera_pos);
        ray r(camera_pos, direction);
        
        color c_sample = sample_mode == 0 ? ray_color_cuda(r, d_linear_nodes, d_scene, d_lights, d_materials, sample_lights, d_pcg_states[idx])
            : ray_color_area_cuda(r, d_linear_nodes, d_scene, d_lights, d_materials, sample_lights, d_pcg_states[idx]);

        float max_rgb = fmaxf(c_sample.r, fmaxf(c_sample.g, c_sample.b));
        if (max_rgb > MAX_SAMPLE_L) {
            float k = MAX_SAMPLE_L / max_rgb;
            c_sample.r *= k;
            c_sample.g *= k;
            c_sample.b *= k;
        }

        double3 c_sample_double;
        c_sample_double.x = fminf(c_sample.r, MAX_SAMPLE_L);
        c_sample_double.y = fminf(c_sample.g, MAX_SAMPLE_L);
        c_sample_double.z = fminf(c_sample.b, MAX_SAMPLE_L);

        d_color_buffer[idx].x = (1.0 / (num_samples + 1.0)) * (num_samples * d_color_buffer[idx].x + c_sample_double.x);
        d_color_buffer[idx].y = (1.0 / (num_samples + 1.0)) * (num_samples * d_color_buffer[idx].y + c_sample_double.y);
        d_color_buffer[idx].z = (1.0 / (num_samples + 1.0)) * (num_samples * d_color_buffer[idx].z + c_sample_double.z);
        // d_color_buffer[idx] = (1.f / (num_samples + 1.f)) * ((float)num_samples * d_color_buffer[idx] + c_sample_double);

        d_image_data[idx].x = static_cast<unsigned char>(fmin(1.0, d_color_buffer[idx].x) * 255.999);
        d_image_data[idx].y = static_cast<unsigned char>(fmin(1.0, d_color_buffer[idx].y) * 255.999);
        d_image_data[idx].z = static_cast<unsigned char>(fmin(1.0, d_color_buffer[idx].z) * 255.999);
        d_image_data[idx].w = 255;
    }
}

__global__ void render_kernel(
    int num_samples,
    int sample_mode,
    bool sample_lights,
    uint width,
    uint height,
    linear_bvh_node* d_linear_nodes,
    size_t h_linear_nodes_bytes,
    char* d_scene_data,
    size_t h_total_scene_bytes,
    char* d_lights_data,
    size_t h_total_lights_bytes,
    char* d_materials_data,
    size_t h_total_materials_bytes,
    double3* d_color_buffer,
    uchar4* d_image_data,
    uint64_t* d_pcg_states
) {
    extern __shared__ char buffer[];

    // buffer layout for shared memory
    // total_scene_bytes d_scene_len * 8 total_lights_bytes d_lights_len * 8    
    // [all shape data | shape pointers | all light data | light pointers | all material data | material pointers]
    linear_bvh_node* shared_linear_nodes = (linear_bvh_node*)buffer;
    char* cur_shape_data = buffer + h_linear_nodes_bytes;
    shape** shared_scene = (shape**)(cur_shape_data + h_total_scene_bytes);
    char* cur_light_data = (char*)(shared_scene + d_scene_len);
    light** shared_lights = (light**)(cur_light_data + h_total_lights_bytes);
    char* cur_material_data = (char*)(shared_lights + d_lights_len);
    material** shared_materials = (material**)(cur_material_data + h_total_materials_bytes);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        memcpy(shared_linear_nodes, d_linear_nodes, h_linear_nodes_bytes);
        /*for (uint i = 0; i < h_linear_nodes_bytes / sizeof(linear_bvh_node); i++) {
            linear_bvh_node& linear_node = shared_linear_nodes[i];
            if (linear_node.is_leaf()) {
                printf("<GPU>[LINEAR LEAF] L %d | R %d", linear_node.L, linear_node.R);
            } else {
                printf("<GPU>[LINEAR INTERIOR] split_axis %d | second_child_idx %d", linear_node.split_axis, linear_node.second_child_idx);
            }
        }*/

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
        point4 camera_pos = d_camera_transform.get_pos();
        float row_offset = randf_pcg32(-0.5f, 0.5f, d_pcg_states[idx]);
        float col_offset = randf_pcg32(-0.5f, 0.5f, d_pcg_states[idx]);
        
        point4 sample = d_pixel00_center + ((col + col_offset) * d_pixel_x) + ((row + row_offset) * d_pixel_y);
        vec4 direction = vec4::normalize(sample - camera_pos);
        ray r(camera_pos, direction);
        
        color c_sample = sample_mode == 0 ? ray_color_cuda(r, shared_linear_nodes, shared_scene, shared_lights, shared_materials, sample_lights, d_pcg_states[idx])
            : ray_color_area_cuda(r, shared_linear_nodes, shared_scene, shared_lights, shared_materials, sample_lights, d_pcg_states[idx]);

        float max_rgb = fmaxf(c_sample.r, fmaxf(c_sample.g, c_sample.b));
        if (max_rgb > MAX_SAMPLE_L) {
            float k = MAX_SAMPLE_L / max_rgb;
            c_sample.r *= k;
            c_sample.g *= k;
            c_sample.b *= k;
        }

        double3 c_sample_double;
        c_sample_double.x = fminf(c_sample.r, MAX_SAMPLE_L);
        c_sample_double.y = fminf(c_sample.g, MAX_SAMPLE_L);
        c_sample_double.z = fminf(c_sample.b, MAX_SAMPLE_L);

        d_color_buffer[idx].x = (1.0 / (num_samples + 1.0)) * (num_samples * d_color_buffer[idx].x + c_sample_double.x);
        d_color_buffer[idx].y = (1.0 / (num_samples + 1.0)) * (num_samples * d_color_buffer[idx].y + c_sample_double.y);
        d_color_buffer[idx].z = (1.0 / (num_samples + 1.0)) * (num_samples * d_color_buffer[idx].z + c_sample_double.z);
        // d_color_buffer[idx] = (1.f / (num_samples + 1.f)) * ((float)num_samples * d_color_buffer[idx] + c_sample_double);

        d_image_data[idx].x = static_cast<unsigned char>(fmin(1.0, d_color_buffer[idx].x) * 255.999);
        d_image_data[idx].y = static_cast<unsigned char>(fmin(1.0, d_color_buffer[idx].y) * 255.999);
        d_image_data[idx].z = static_cast<unsigned char>(fmin(1.0, d_color_buffer[idx].z) * 255.999);
        d_image_data[idx].w = 255;
    }
}

__global__ void init_pcg_states_kernel(uint64_t* d_pcg_states, uint width, uint height) {
    uint row = blockDim.x * blockIdx.x + threadIdx.x;
    uint col = blockDim.y * blockIdx.y + threadIdx.y;
    uint idx = row * width + col;

    if (row < height && col < width) {
        d_pcg_states[idx] = 0x4d595df4d0f33173;
        pcg32_init(idx, d_pcg_states[idx]);
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

void export_image(uchar4* d_image_data, uint num_pixels) {
    std::vector<uchar4> image_data(num_pixels);
    gpuErrchk(cudaMemcpy(image_data.data(), d_image_data, num_pixels * sizeof(uchar4), cudaMemcpyDeviceToHost));
    std::ofstream file{ "renders/" + time_stamp() + ".ppm", std::ios::app };
    file << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (uint row = 0; row < image_height; row++) {
        std::clog << "\rScanlines remaining: " << (image_height - row) << "     " << std::flush;
        for (uint col = 0; col < image_width; col++) {
            uint idx = row * image_width + col;
            file << (int)image_data[idx].x << " " << (int)image_data[idx].y << " " << (int)image_data[idx].z << " ";
        }
    }
    
    std::clog << "\rEXPORT COMPLETE                                                    \n";
    file.close();
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
}

#endif