#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.h"

struct dielectric : public material {
    float eta;

    __host__ __device__ dielectric(): eta{1.f} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        return {0.f, 0.f, 0.f};
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        bs.f = color{0.f, 0.f, 0.f}; //* inv_pi;
        
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