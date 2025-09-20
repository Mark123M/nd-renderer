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

    __host__ __device__ bool is_specular() const override {
        return true;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(specular);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Specular | Albedo (%.3f, %.3f, %.3f)\n", albedo.r, albedo.g, albedo.b);
    }
};

struct rough_specular : public material {
    color albedo;
    float alpha;

    __host__ __device__ rough_specular(): albedo{} {}

    __host__ __device__ rough_specular(const color& albedo): albedo{albedo} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        if (!same_side(wo, wi)) {
            return {0.f, 0.f, 0.f};
        }

        return albedo; //* inv_pi;
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        vec4 wm = vec4::rand_halfsphere_vector_cosine_weighted(pcg_state);
        bs.pdf = (2.f * wm.y) / (pi * pi);
        bs.wi = vec4::reflect(wo, wm);
        
        if (!same_side(wo, bs.wi)) {
            bs.f = color(0.f, 0.f, 0.f);
            return true;
        }

        float jacobian = 1.f / (4.f * fabsf(vec4::dot(wo, wm)));
        bs.pdf = bs.pdf * jacobian;
        float cos_wo_wm = fabsf(vec4::dot(wo, wm)); // wo is incident on the microfacet
        color F0 = albedo;
        color pr = F0 + (color(1.f) - F0) * powf((1.f - cos_wo_wm), 5.f);

        float cos_theta_o = fabsf(wo.y);
        float cos_theta_i = fabsf(bs.wi.y);
        bs.f = pr * GGX::D(wm, alpha) * GGX::G(wo, bs.wi, alpha) / (4.f * cos_theta_i * cos_theta_o);
        return true;
    }

    __host__ __device__ bool is_specular() const override {
        return alpha <= 0.001f;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(rough_specular);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Specular | Albedo (%.3f, %.3f, %.3f)\n", albedo.r, albedo.g, albedo.b);
    }
};

#endif