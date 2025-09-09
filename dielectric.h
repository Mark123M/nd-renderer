#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.h"
#include "util.h"

struct dielectric : public material {
    float eta;

    __host__ __device__ dielectric(float eta0 = 1.f): eta{eta0} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        return {0.f, 0.f, 0.f};
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        bs.f = color{1.f, 1.f, 1.f}; //* inv_pi;
        float cos_theta = wo.y;
        float etar = eta;
        vec4 normal(0.f, 1.f, 0.f, 0.f);

        if (cos_theta < 0.f) {
			etar = 1.f / etar;
            cos_theta = -cos_theta;
            normal.y = -1.f;
        }

        float r0 = (1.f - etar) / (1.f + etar);
        r0 = r0 * r0;
        float fresnel = r0 + (1.f - r0) * powf((1.f - cos_theta), 5.f);

        if (randf_pcg32(0.f, 1.f, pcg_state) < fresnel) {
            bs.wi = vec4::reflect(wo); // might be wrong
        } else {
            float sin2_theta_i = fmaxf(0.f, 1.f - cos_theta * cos_theta);
            float sin2_theta_t = sin2_theta_i / (etar * etar);
            if (sin2_theta_t >= 1.f) {
                bs.wi = vec4::reflect(wo);
            } else {
                float cos_theta_t = sqrtf(1.f - sin2_theta_t);

                // Standard vector refraction formula: R = η_rel * I + (η_rel * cos(i) - cos(t)) * N
                // Where our I is -wo.
                bs.wi = -wo / etar + (cos_theta / etar - cos_theta_t) * normal;
            }
        }

        return true;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(dielectric);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Dielectric | eta %.3f", eta);
    }
};

#endif