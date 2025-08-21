#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.h"

struct lambertian : public material {
    color albedo;

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        if (!same_side(wo, wi)) {
            return {0.f, 0.f, 0.f};
        }

        return albedo * inv_pi;
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        bs.f = albedo * inv_pi;
        bs.wi = vec4::rand_halfsphere_vector(pcg_state);
        return true;
    }

    __host__ __device__ size_t size() const {
        return sizeof(lambertian);
    }
}

#endif