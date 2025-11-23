#ifndef AABB_H
#define AABB_H

#include "vec4.h"
#include <string>
#include <format>

struct aabb {
    point4 p_min, p_max;
    // empty box
    __host__ __device__ aabb(): p_min{POINT4_MAX}, p_max{POINT4_MIN} {}
    __host__ __device__ aabb(const point4& p1, const point4& p2): p_min{point4::min(p1, p2)}, p_max{point4::max(p1, p2)} {}

    __host__ __device__ bool intersect(const ray& r, float t_max) const {
        float t0 = 0, t1 = t_max;
        
        for (uint i = 0; i < 4; i++) {
            float inv_dir = 1.f / r.dir.get(i);
            float t_near = (p_min.get(i) - r.pos.get(i)) * inv_dir;
            float t_far  = (p_max.get(i) - r.pos.get(i)) * inv_dir;

            if (t_near > t_far) {
                float tmp = t_near;
                t_near = t_far;
                t_far = tmp;
                //std::swap(t_near, t_far);
            }

            t_far *= 1 + EPSILON;

            t0 = fmaxf(t0, t_near);
            t1 = fminf(t1, t_far);
            if (t0 > t1) {
                return false;
            }
        }

        return t1 > 0.f;
    }

    __host__ __device__ float surface_volume() const {
        vec4 lengths = p_max - p_min;
        return 2.f * (lengths.x * lengths.y * lengths.z +
                      lengths.x * lengths.y * lengths.w +
                      lengths.x * lengths.z * lengths.w +
                      lengths.y * lengths.z * lengths.w);
    }

    __host__ __device__ uint max_dim() const {
        vec4 lengths = p_max - p_min;
        uint dim = 0;
        float max_len = lengths.get(0);

        for (uint i = 1; i < 4; i++) {
            if (lengths.get(i) > max_len) {
                dim = i;
                max_len = lengths.get(i);
            }
        }
        
        return dim;
    }

    __host__ __device__ point4 centroid() const {
        return 0.5f * p_min + 0.5f * p_max;
    }

    __host__ __device__ static aabb merge(const aabb& b1, const aabb& b2) {
        aabb b;
        b.p_min = point4::min(b1.p_min, b2.p_min);
        b.p_max = point4::max(b1.p_max, b2.p_max);
        return b;
    }

    __host__ __device__ static aabb merge(const aabb& b0, const point4& p) {
        aabb b;
        b.p_min = point4::min(b0.p_min, p);
        b.p_max = point4::max(b0.p_max, p);
        return b;
    }
};

#endif