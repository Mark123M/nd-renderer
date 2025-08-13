#ifndef DIRECTION_LIGHT_H
#define DIRECTION_LIGHT_H

#include "vec4.h"
#include "light.h"
#include "camera.h"

struct direction_light : public light {
	vec4 dir;

	__host__ __device__ direction_light() : light{}, dir{0.f, 0.f, -1.f, 0.f} {}

	__host__ color Le(point4 p, vec4 normal) override {
		float diffuse = fmaxf(0.f, vec4::dot(normal, dir));
        point4 shadow_ray_origin = p + normal * 0.01f; // slight offset
        ray shadow_ray(shadow_ray_origin, dir);
        shape* shadow_target = nullptr;
        camera::ray_march(shadow_ray, &shadow_target);

        bool visibility = shadow_target == nullptr; // check of occlusion
        return (diffuse * visibility + AMBIENT) * col;
	}

	__device__ color Le_cuda(point4 p, vec4 normal, shape** shared_scene, light** shared_lights) override {
		float diffuse = fmaxf(0.f, vec4::dot(normal, dir));
        point4 shadow_ray_origin = p + normal * 0.01f; // slight offset
        ray shadow_ray(shadow_ray_origin, dir);
        shape* shadow_target = nullptr;
        camera::ray_march_cuda(shadow_ray, &shadow_target, shared_scene, shared_lights);

        bool visibility = shadow_target == nullptr; // check of occlusion
        return (diffuse * visibility + AMBIENT) * col;
	}

    __host__ __device__ size_t size() const override {
        return sizeof(direction_light);
    }

    __device__ virtual void print_gpu() const override {
        printf("[GPU] Direction Light | Direction (%.3f, %.3f, %.3f, %.3f) | Color (%.3f, %.3f, %.3f)\n",
        dir.x, dir.y, dir.z, dir.w, col.r, col.g, col.b);
    }
};

#endif // !DIRECTION_LIGHT_H
