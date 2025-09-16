#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.h"
#include "util.h"

struct smooth_dielectric : public material {
    float eta;

    __host__ __device__ smooth_dielectric(float eta0 = 1.f): eta{eta0} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        return {0.f, 0.f, 0.f};
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        bs.pdf = 1.f;
        float cos_theta = wo.y;
        float etar = eta;
        vec4 normal(0.f, 1.f, 0.f, 0.f);

        if (cos_theta < 0.f) {
            cos_theta = -cos_theta;
			etar = 1.f / etar;
            normal.y = -1.f;
        }

        float pr;
        float sin2_theta_i = fmaxf(0.f, 1.f - cos_theta * cos_theta);
        float sin2_theta_t = sin2_theta_i / (etar * etar);
        
        if (sin2_theta_t >= 1.f) { // total internal reflection
            pr = 1.f;
        } else {
            float r0 = (1.f - etar) / (1.f + etar);
            r0 = r0 * r0;
            pr = r0 + (1.f - r0) * powf((1.f - cos_theta), 5.f);
        }

        if (randf_pcg32(0.f, 1.f, pcg_state) < pr) {
            bs.wi = vec4::reflect(wo);
            bs.f = color(1.f, 1.f, 1.f) / cos_theta;
        } else {
            float cos_theta_t = sqrtf(1.f - sin2_theta_t);
            // Standard vector refraction formula: R = η_rel * I + (η_rel * cos(i) - cos(t)) * N
            // Where our I is -wo.
            bs.wi = -wo / etar + (cos_theta / etar - cos_theta_t) * normal;
            float eta_scale = 1.f / (etar * etar);
            bs.f = color(1.f, 1.f, 1.f) * eta_scale / cos_theta;
        }

        return true;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(smooth_dielectric);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Smooth dielectric | eta %.3f", eta);
    }
};

struct rough_dielectric : public material {
    float eta;
    float alpha;

    __host__ __device__ rough_dielectric(float eta0 = 1.f): eta{eta0} {}

    __host__ __device__ color f(const vec4& wo, const vec4& wi) const override {
        return {0.f, 0.f, 0.f};
    }

    __host__ __device__ bool sample_f(const vec4& wo, bsdf_sample& bs, uint64_t& pcg_state) const override {
        bs.pdf = 1.f / (pi * pi); // sampling half 3-sphere
        vec4 wm = vec4::rand_halfsphere_vector(pcg_state);
        float cos_wo_wm = vec4::dot(wo, wm);
        float etar = eta;
        // 1. Importance sample the microfacet normal wm
        /*float u1 = randf_pcg32(0.f, 1.f, pcg_state);
        float u2 = randf_pcg32(0.f, 1.f, pcg_state);
        float a2 = alpha * alpha;
        float phi_m = 2.f * pi * u1;
        float cos_theta_m = sqrtf((1.f - u2) / (1.f + (a2 - 1.f) * u2));
        float sin_theta_m = sqrtf(fmaxf(0.f, 1.f - cos_theta_m * cos_theta_m));

        vec4 wm(
            sin_theta_m * cosf(phi_m),
            cos_theta_m,
            sin_theta_m * sinf(phi_m),
            0.f
        );
        vec4::normalize(wm); */

        if (cos_wo_wm < 0.f) {
            cos_wo_wm = -cos_wo_wm;
            etar = 1.f / etar;
            wm = -wm;
        }

        float pr, pt;
        float sin2_theta_i = fmaxf(0.f, 1.f - cos_wo_wm * cos_wo_wm);
        float sin2_theta_t = sin2_theta_i / (etar * etar);

        if (sin2_theta_t >= 1.f) {
            pr = 1.f;
        } else {
            float r0 = (1.f - etar) / (1.f + etar);
            r0 = r0 * r0;
            pr = r0 + (1.f - r0) * powf((1.f - cos_wo_wm), 5.f);
        }

        pt = 1.f - pr;

        if (randf_pcg32(0.f, 1.f, pcg_state) < pr) {
            bs.wi = vec4::reflect(wo, wm);
            // get pdf
            bs.f = GGX::D(wm, alpha) * GGX::G(wo, bs.wi, alpha) * pr / (4.f * bs.wi.y * wo.y); // may need to flip cos theta in denom?
        } else {
            float cos_theta_t = sqrtf(1.f - sin2_theta_t);

            // Standard vector refraction formula: R = η_rel * I + (η_rel * cos(i) - cos(t)) * N
            // Where our I is -wo.
            bs.wi = -wo / etar + (cos_wo_wm / etar - cos_theta_t) * wm;

            float d = vec4::dot(bs.wi, wm) + vec4::dot(wo, wm) / etar;
            float denom = d * d;
            float dwm_dwi = fabsf(vec4::dot(bs.wi, wm)) / denom;

            // get pdf
            bs.f = color(pt * GGX::D(wm, alpha) * GGX::G(wo, bs.wi, alpha) * fabsf(vec4::dot(bs.wi, wm) * vec4::dot(wo, wm) / (bs.wi.y * wo.y * denom)));
            bs.f /= etar * etar; // // Account for non-symmetry with transmission to different medium
        }

        return true;
    }

    __host__ __device__ size_t size() const override {
        return sizeof(rough_dielectric);
    }

    __device__ void print_gpu() const override {
        printf("[GPU] Rough dielectric | eta %.3f", eta);
    }
};

#endif