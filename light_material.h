#ifndef LIGHT_MATERIAL_H
#define LIGHT_MATERIAL_H

#include "material.h"

struct light_material : public material {
    color col;

    __host__ __device__ light_material(): col{} {}

    __host__ __device__ light_material(const color& col): col{col} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        return color(0.f, 0.f, 0.f);
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        bs.f = col; //* inv_pi;
        bs.wi = vec4(0.f, 0.f, 0.f, 0.f);
        return false;
    }

    __host__ __device__ color L() const override {
        return col;
    }

    __host__ __device__ bool sample_li(const shape* s, const hit_result& res, light_sample& ls, uint64_t& pcg_state) const override {
        shape_sample ss;
        if (!s->sample(ss, res, pcg_state)) { // WRONG!!! sample light shape
            return false;
        }
        vec4 wi = vec4::normalize(ss.p - res.p);
        ls.L = col;
        ls.wi = wi;
        ls.pdf = ss.pdf;
        return true;
    }

    __host__ __device__ bool is_specular() const override {
        return false;
    }

    __host__ __device__ bool can_sample_f() const override {
        return false;
    }

    __host__ __device__ bool is_emissive() const override {
        return true;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(light_material);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Light material | Color (%.3f, %.3f, %.3f)\n", col.r, col.g, col.b);
    }
};

#endif