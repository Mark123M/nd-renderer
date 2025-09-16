#ifndef SPECULAR_H
#define SPECULAR_H

#include "material.h"

struct specular : public material {
    color albedo;

    __host__ __device__ specular(): albedo{} {}

    __host__ __device__ specular(const color& albedo): albedo{albedo} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        if (!same_side(wo, wi)) {
            return {0.f, 0.f, 0.f};
        }

        return albedo; //* inv_pi;
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        float cos_theta = fabsf(wo.y);
        if (cos_theta < 1e-6f) {
            bs.f = color(0.f, 0.f, 0.f);
            return false; // Or handle appropriately
        }

        // Pre-divide the reflectance by the cosine term.
        bs.f = albedo / cos_theta;
        bs.wi = vec4::reflect(wo);
        bs.pdf = 1.f; // must reflect
        return true;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(specular);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Specular | Albedo (%.3f, %.3f, %.3f)\n", albedo.r, albedo.g, albedo.b);
    }
};

#endif