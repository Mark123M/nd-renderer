#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "transform.h"

struct bsdf_sample {
	color f;
	vec4 wi;
	float pdf = 0;
	float eta = 1;
};

enum material_type {
    LAMBERTIAN, SPECULAR, DIELECTRIC
};

struct material {
    //int flags = 0;
    __host__ __device__ virtual color f(const vec4& wo, const vec4& wi) const = 0;
    __host__ __device__ virtual bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const = 0;
    __host__ __device__ virtual size_t size() const = 0;
    __device__ virtual void print_gpu() const = 0;

    __host__ __device__ color f(const vec4& wo_world, const vec4& wi_world, const transform& t) const {
        vec4 wo = t.world_to_local(wo_world);
        vec4 wi = t.world_to_local(wi_world);
        return f(wo, wi);
    }

    __host__ __device__ bool sample_f(const vec4& wo_world, const transform& t, bsdf_sample& bs, uint64_t& pcg_state) const {
        vec4 wo = t.world_to_local(wo_world);

        if (!sample_f(wo, bs, pcg_state)) {
            return false;
        }

        bs.wi = vec4::normalize(t.local_to_world(bs.wi)); // re-normalize for for accumulated fp errors
        return true;
    }

    __host__ __device__ bool same_side(const vec4& u, const vec4& v) const {
        return u.y * v.y >= 0;
    }
};

namespace GGX {
    __host__ __device__ float lambda(const vec4& w, float alpha) {
        float cos2_theta = vec4::cos2_theta(w);
        if (cos2_theta < 1e-6f) return 0.f; // Avoid division by zero

        float tan2_theta = vec4::tan2_theta(w);
        return (sqrtf(1.f + alpha * alpha * tan2_theta) - 1.f) / 2.f;
    }

    __host__ __device__ float G1(const vec4& w, float alpha) {
        return 1.f / (1.f + lambda(w, alpha));
    }
    
    __host__ __device__ float G(const vec4& wo, const vec4& wi, float alpha) {
        return 1.f / (1.f + lambda(wo, alpha) + lambda(wi, alpha));
    }

    __host__ __device__ float D(const vec4& wm, float alpha) {
        float cos2_theta = vec4::cos2_theta(wm);
        if (cos2_theta < 1e-6f) return 0.f; // Avoid division by zero

        float a2 = alpha * alpha;
        float tan2_theta = vec4::tan2_theta(wm);
        float denom = pi * a2 * cos2_theta * cos2_theta * (1.f + tan2_theta / a2) * (1.f + tan2_theta / a2); 
        return 1.f / denom;
    }

    __host__ __device__ float D(const vec4& w, const vec4& wm, float alpha) {
        return G1(w, alpha) / fabsf(w.y) * D(wm, alpha) * fabsf(vec4::dot(w, wm));
    }
};

#endif