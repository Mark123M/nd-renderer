#ifndef CAMERA_H
#define CAMERA_H

#include <sstream>
#include <vector>
#include <ctime>
#include <mutex>

#include "color.h"
#include "ray.h"
#include "math_util.h"
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

std::vector<shape*> scene;
__device__ shape** d_scene;
__constant__ size_t d_scene_len;
std::vector<light*> lights;
__device__ light** d_lights;
__constant__ size_t d_lights_len;
std::vector<uchar4> image_data;

namespace camera {

int image_width, image_height;
float aspect_ratio = 1.f;
float focal_length = 1.f;
bool render_normals = false;
__constant__ bool d_render_normals;

vec4 pixel_x;
__constant__ vec4 d_pixel_x;
vec4 pixel_y;
__constant__ vec4 d_pixel_y;
point4 camera_center;
__constant__ point4 d_camera_center;
point4 pixel00_center;
__constant__ point4 d_pixel00_center;
//point4 lookfrom = point4(0.f, 0.f, 0.f, 0.f);
//point4 lookat = point4(0.f, 0.f, -1.f, 0.f);

__device__ float scene_sdf_cuda(const point4& p, shape** target_ptr) {
    float sdf = MAX_DIST + 5;
    
    for (size_t i = 0; i < d_scene_len; i++) {
        shape* obj = d_scene[i];
        float obj_sdf = obj->sdf(p);

        if (obj_sdf < sdf) {
            sdf = obj_sdf;
            *target_ptr = obj; // assign object pointer
        }
    }

    assert(sdf >= -TOL); // only reflections
    return sdf;
}

__device__ float scene_sdf_cuda(const point4& p) {
    float sdf = MAX_DIST + 5;

    for (size_t i = 0; i < d_scene_len; i++) {
        shape* obj = d_scene[i];
        float obj_sdf = obj->sdf(p);

        if (obj_sdf < sdf) {
            sdf = obj_sdf;
        }
    }

    assert(sdf >= -TOL); // only reflections
    return sdf;
}

__device__ vec4 get_normal_cuda(const vec4& p) {
    float sdf_diff_x = scene_sdf_cuda(p + d_delta_x) - scene_sdf_cuda(p - d_delta_x);
    float sdf_diff_y = scene_sdf_cuda(p + d_delta_y) - scene_sdf_cuda(p - d_delta_y);
    float sdf_diff_z = scene_sdf_cuda(p + d_delta_z) - scene_sdf_cuda(p - d_delta_z);
    float sdf_diff_w = scene_sdf_cuda(p + d_delta_w) - scene_sdf_cuda(p - d_delta_w);

    return vec4::normalize(vec4(sdf_diff_x, sdf_diff_y, sdf_diff_z, sdf_diff_w));
}

__device__ void ray_march_cuda(ray& r, shape** target_ptr) {
    //float a = 0.5f * (r.dir.y + 1.f);
    //return (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
    do {
        float dist = scene_sdf_cuda(r.pos, target_ptr);

        if (dist <= TOL) {
            return;
        } else if (dist > MAX_DIST) {
            *target_ptr = nullptr; // no object hit
            return;
        } else {
            r.march(dist);
        }
    } while (true);
}

__device__ color ray_color_cuda(ray& r) {
    shape* target = nullptr;
    ray_march_cuda(r, &target);

    if (target == nullptr) {
        vec4 unit_direction = vec4::normalize(r.dir);
        float a = 0.5f * (unit_direction.y + 1.f);
        return (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
    }

    vec4 normal = get_normal_cuda(r.pos);

    if (d_render_normals) {
        return 0.5 * color(normal.x + 1, normal.y + 1, normal.z + 1);
    }

    color total_lighting(0.f, 0.f, 0.f);

    for (size_t i = 0; i < d_lights_len; i++) {
        light* lig = d_lights[i];
        total_lighting += lig->Le_cuda(r.pos, normal);
        //printf("[GPU] Hit pos (%.3f, %.3f, %.3f, %.3f) | Total lighting (%.3f, %.3f, %.3f)\n",
        //r.pos.x, r.pos.y, r.pos.z, r.pos.w, total_lighting.r, total_lighting.g, total_lighting.b);
    }

    color final_col = target->albedo * total_lighting;

    return final_col;
}

__global__ void render_kernel(uchar4* d_image_data, int width, int height) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = row * width + col;

    if (row < height && col < width) {
        // need device versions of these functions/structs
        point4 pixel_center = d_pixel00_center + (col * d_pixel_x) + (row * d_pixel_y);
        vec4 direction = vec4::normalize(pixel_center - d_camera_center);
        ray r(d_camera_center, direction);
        
        color color = ray_color_cuda(r);
        d_image_data[idx].x = static_cast<unsigned char>(fminf(1.f, color.r) * 255.999f);
        d_image_data[idx].y = static_cast<unsigned char>(fminf(1.f, color.g) * 255.999f);
        d_image_data[idx].z = static_cast<unsigned char>(fminf(1.f, color.b) * 255.999f);
        d_image_data[idx].w = 255;
    }
}

float scene_sdf(const point4& p, shape** target_ptr) {
    float sdf = MAX_DIST + 5;
    
    for (shape* obj : scene) {
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
    float sdf = MAX_DIST + 5;

    for (shape* obj : scene) {
        float obj_sdf = obj->sdf(p);

        if (obj_sdf < sdf) {
            sdf = obj_sdf;
        }
    }

    assert(sdf >= -TOL); // only reflections
    return sdf;
}

vec4 get_normal(const vec4& p) {
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
        } else if (dist > MAX_DIST) {
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
        return 0.5 * color(normal.x + 1, normal.y + 1, normal.z + 1);
    }

    color total_lighting(0.f, 0.f, 0.f);

    for (light* lig : lights) {
        total_lighting += lig->Le(r.pos, normal);
        //printf("[CPU] Hit pos (%.3f, %.3f, %.3f, %.3f) | Total lighting (%.3f, %.3f, %.3f)\n",
        //r.pos.x, r.pos.y, r.pos.z, r.pos.w, total_lighting.r, total_lighting.g, total_lighting.b);
    }

    color final_col = target->albedo * total_lighting;

    return final_col;
}

void initialize() {
    image_height = std::max(1.f, image_width / aspect_ratio);

    float viewport_height = 2.f;
    float viewport_width = viewport_height * ((float)image_width / image_height);
    camera_center = point4(0.f, 0.f, 1.f, 0.f);

    vec4 viewport_x = vec4(viewport_width, 0.f, 0.f, 0.f);
    vec4 viewport_y = vec4(0.f, -viewport_height, 0.f, 0.f);
    pixel_x = viewport_x / image_width;
    pixel_y = viewport_y / image_height;

    point4 viewport_top_left = camera_center - vec4(0.f, 0.f, focal_length, 0.f) - viewport_x / 2 - viewport_y / 2;
    pixel00_center = viewport_top_left + 0.5 * (pixel_x + pixel_y);

    gpuErrchk(cudaMemcpyToSymbol(d_render_normals, &render_normals, sizeof(bool)));
    gpuErrchk(cudaMemcpyToSymbol(d_pixel_x, &pixel_x, sizeof(vec4)));
    gpuErrchk(cudaMemcpyToSymbol(d_pixel_y, &pixel_y, sizeof(vec4)));
    gpuErrchk(cudaMemcpyToSymbol(d_camera_center, &camera_center, sizeof(point4)));
    gpuErrchk(cudaMemcpyToSymbol(d_pixel00_center, &pixel00_center, sizeof(point4)));

    gpuErrchk(cudaMemcpyToSymbol(d_delta_x, &delta_x, sizeof(vec4)));
    gpuErrchk(cudaMemcpyToSymbol(d_delta_y, &delta_y, sizeof(vec4)));
    gpuErrchk(cudaMemcpyToSymbol(d_delta_z, &delta_z, sizeof(vec4)));
    gpuErrchk(cudaMemcpyToSymbol(d_delta_w, &delta_w, sizeof(vec4)));
}

void render_rt(int first_row, int last_row) {
    // assume camera parameters are all initialized
    for (int row = first_row; row <= last_row; row++) {
        //std::clog << "\rScanlines remaining: " << (image_height - j) << "     " << std::flush;
        for (int col = 0; col < image_width; col++) {
            int idx = row * image_width + col;
            point4 pixel_center = pixel00_center + (col * pixel_x) + (row * pixel_y);
            vec4 direction = vec4::normalize(pixel_center - camera_center);
            ray r(camera_center, direction);
            
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

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << "     " << std::flush;
        for (int i = 0; i < image_width; i++) {
            point4 pixel_center = pixel00_center + (i * pixel_x) + (j * pixel_y);
            vec4 direction = vec4::normalize(pixel_center - camera_center);
            ray r(camera_center, direction);

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