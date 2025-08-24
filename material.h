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
    LAMBERTIAN, SPECULAR, GLASS
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

        bs.wi = t.local_to_world(bs.wi);
        return true;
    }

    __host__ __device__ bool same_side(const vec4& u, const vec4& v) const {
        return u.y * v.y >= 0;
    }
};

#endif