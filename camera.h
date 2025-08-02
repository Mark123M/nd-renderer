#ifndef CAMERA_H
#define CAMERA_H

#include "shape.h"
#include "color.h"
#include <sstream>
#include "ray.h"
#include <vector>
#include <cassert>
#include "util.h"
#include "direction_light.h"
#include <ctime>
#include <mutex>

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

namespace camera {

int image_width, image_height;
float aspect_ratio = 1.f;
float focal_length = 1.f;
bool render_normals = false;

vec4 pixel_x;
vec4 pixel_y;
point4 camera_center;
point4 pixel00_center;
//point4 lookfrom = point4(0.f, 0.f, 0.f, 0.f);
//point4 lookat = point4(0.f, 0.f, -1.f, 0.f);

std::vector<shape*> scene;
std::vector<direction_light> lights;
std::vector<unsigned char> image_data;

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

    for (direction_light& light : lights) {
        float diffuse = std::max(0.f, vec4::dot(normal, light.dir));
        point4 shadow_ray_origin = r.pos + normal * 0.01; // slight offset
        ray shadow_ray(shadow_ray_origin, light.dir);
        shape* shadow_target = nullptr;
        ray_march(shadow_ray, &shadow_target);

        bool visibility = shadow_target == nullptr; // check of occlusion
        total_lighting += (diffuse * visibility + AMBIENT) * light.col;
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
    
    image_data = std::vector<unsigned char>(image_width * image_height * 3);
}

void render_rt(int first_row, int last_row) {
    // assume camera parameters are all initialized
    for (int row = first_row; row <= last_row; row++) {
        //std::clog << "\rScanlines remaining: " << (image_height - j) << "     " << std::flush;
        for (int col = 0; col < image_width; col++) {
            point4 pixel_center = pixel00_center + (col * pixel_x) + (row * pixel_y);
            vec4 direction = vec4::normalize(pixel_center - camera_center);
            ray r(camera_center, direction);
            
            color color = ray_color(r);
            int idx = row * image_width * 3 + col * 3;
            image_data[idx + 0] = static_cast<unsigned char>(std::min(1.f, color.r) * 255.999f);
            image_data[idx + 1] = static_cast<unsigned char>(std::min(1.f, color.g) * 255.999f);
            image_data[idx + 2] = static_cast<unsigned char>(std::min(1.f, color.b) * 255.999f);
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