#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.h"

struct lambertian : public material {
    color albedo;

    __host__ __device__ lambertian(): albedo{} {}

    __host__ __device__ lambertian(const color& albedo): albedo{albedo} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        // if (wo.y < 0.f) {
        //    return color(0.f, 0.f, 0.f);
        // }

        return 0.75f * inv_pi * albedo; // (74.2f / (100.f * pi)) * albedo; //* inv_pi;
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        bs.f = 0.75f * inv_pi * albedo; //* inv_pi;
        bs.wi = vec4::rand_halfsphere_vector(pcg_state);
        bs.pdf = 1.f / (pi * pi);

        if (wo.y < 0.f) {
            bs.wi = -bs.wi; // flip sampling
        }
        
        return true;
    }

    __host__ __device__ bool is_specular() const override {
        return false;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(lambertian);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Lambertian | Albedo (%.3f, %.3f, %.3f)\n", albedo.r, albedo.g, albedo.b);
    }
};

#endif