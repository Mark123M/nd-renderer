#ifndef CUBE_H
#define CUBE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

// Represents a quadrilateral (specifically, a parallelogram) shape.
// It is defined by a corner point `q`, and two edge vectors, `u` and `v`.
struct cube : public shape {
	float half_len;
    float half_w;
    
	__host__ __device__ cube(): half_len{ 0.25f }, half_w{ TOL } {}

	__host__ __device__ bool intersect(const ray& r, hit_result& res) const override {
        point4 local_r_pos = basis.world_to_local(r.pos);
        vec4   local_r_dir = basis.world_to_local(r.dir);

        float t_near = -FLT_MAX;
        float t_far  =  FLT_MAX;
        int near_axis = -1;
        int far_axis = -1;

        // Perform the slab test for all 4 axes (x, y, z, w).
        for (int i = 0; i < 4; ++i) {
            // Use half_len for x,y,z (i<3) and half_w for w (i=3).
            float current_half_extent = (i < 3) ? half_len : half_w;

            float origin_comp = local_r_pos.get(i);
            float dir_comp = local_r_dir.get(i);

            if (fabsf(dir_comp) < 1e-6f) {
                if (origin_comp < -current_half_extent || origin_comp > current_half_extent) {
                    return false;
                }
                continue;
            }

            float t1 = (-current_half_extent - origin_comp) / dir_comp;
            float t2 = ( current_half_extent - origin_comp) / dir_comp;

            if (t1 > t2) {
                float temp = t1; t1 = t2; t2 = temp;
            }

            if (t1 > t_near) {
                t_near = t1;
                near_axis = i;
            }
            if (t2 < t_far) {
                t_far = t2;
                far_axis = i;
            }
        }

        if (t_near >= t_far || t_far < TOL) {
            return false;
        }

        float t_hit = t_near;
        int hit_axis = near_axis;
        bool starts_inside = false;

        if (t_hit < TOL) {
            t_hit = t_far;
            hit_axis = far_axis;
            starts_inside = true;
        }

        if (t_hit >= res.t) {
            return false;
        }

        // The manual check for 'w' is no longer needed, it's handled by the loop.

        vec4 local_normal(0.f, 0.f, 0.f, 0.f);
        if (hit_axis != -1) {
            float sign = starts_inside ? 
                copysignf(1.0f, local_r_pos.get(hit_axis) + local_r_dir.get(hit_axis) * t_hit) :
                -copysignf(1.0f, local_r_dir.get(hit_axis));
            local_normal.set(hit_axis, sign);
        } else {
            return false;
        }

        res.t = t_hit;
        res.p = r.pos + res.t * r.dir;
        res.wo = -r.dir;
        res.target = this;
        res.normal = vec4::normalize(basis.local_to_world(local_normal));
        res.m = transform::get_shading_transform(res.normal);

        return true;
    }

    std::unique_ptr<shape> clone() const override {
        return std::make_unique<cube>(*this);
    }

    // Returns the size of the object in memory.
	__host__ __device__ size_t size() const override {
		return sizeof(cube);
	}
	
    // Prints information about the quad from the GPU.
	__device__ virtual void print_gpu() const override {
		printf("[GPU] Cube | Half len %.3f", half_len);
	}
};

#endif // QUAD_H
