#ifndef LIGHT_MATERIAL_H
#define LIGHT_MATERIAL_H

#include "material.h"

struct light_material : public material {
    color albedo;

    __host__ __device__ light_material(): albedo{} {}

    __host__ __device__ light_material(const color& albedo): albedo{albedo} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        if (!same_side(wo, wi)) {
            return {0.f, 0.f, 0.f};
        }

        return albedo; //* inv_pi;
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        bs.f = albedo; //* inv_pi;
        bs.wi = vec4(0.f, 0.f, 0.f, 0.f);
        return false;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(light_material);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Light material | Albedo (%.3f, %.3f, %.3f)\n", albedo.r, albedo.g, albedo.b);
    }
};

#endif