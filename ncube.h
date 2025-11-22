#ifndef NCUBE_H
#define NCUBE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct ncube : public shape {
	point4 corner;
	float half_len;

	__host__ __device__ ncube() : shape{}, corner{ 0.f, 0.f, 0.f, 0.f }, half_len{ 0.25f } {}

	__host__ __device__ float sdf(const point4& p) const override {
		point4 pp = basis.world_to_local(p); //multiply_matrix_vec4(example_matrix, p);
		
		vec4 q = vec4::abs(pp - origin) - (corner - origin);
		float d = vec4::max(q, 0.f).length() + fminf(vec4::max_comp(q), 0.f);
		return d;
	}

	__host__ __device__ bool intersect(const ray& r, hit_result& res) const override {
        // Step 1: Transform the ray into the tesseract's local coordinate system.
        // The slab test is trivial for an axis-aligned unit hypercube.
        point4 local_r_pos = basis.world_to_local(r.pos);
        vec4   local_r_dir = basis.world_to_local(r.dir);

        float t_near = -FLT_MAX;
        float t_far  =  FLT_MAX;
        int near_axis = -1; // To remember which face we hit
        int far_axis = -1;

        // Step 2: Perform the slab test for each of the 4 axes (x, y, z, w).
        for (int i = 0; i < 4; ++i) {
            float origin_comp = local_r_pos.get(i);
            float dir_comp = local_r_dir.get(i);

            // Check if the ray is parallel to the slab planes for this axis.
            if (fabsf(dir_comp) < 1e-6f) {
                // If the ray is parallel but outside the slab, it can never intersect.
                if (origin_comp < -half_len || origin_comp > half_len) {
                    return false;
                }
                // Otherwise, it doesn't constrain the interval, so we continue.
                continue;
            }

            // Calculate intersection distances for the two hyperplanes of the slab.
            // e.g., for x-axis, the planes are x=-1 and x=1.
            float t1 = (-half_len - origin_comp) / dir_comp;
            float t2 = ( half_len - origin_comp) / dir_comp;

            // Ensure t1 is the smaller value
            if (t1 > t2) {
                float temp = t1; t1 = t2; t2 = temp;
            }

            // Update the overall intersection interval [t_near, t_far].
            if (t1 > t_near) {
                t_near = t1;
                near_axis = i;
            }
            if (t2 < t_far) {
                t_far = t2;
                far_axis = i;
            }
        }

        // Step 3: Check if the ray misses the tesseract.
        // This happens if the entry point is after the exit point, or if the
        // entire intersection interval is behind the ray's origin.
        if (t_near > t_far || t_far < TOL) {
            return false;
        }

        // Step 4: Determine the correct intersection distance 't_hit'.
        float t_hit = t_near;
        int hit_axis = near_axis;
        bool starts_inside = false;

        if (t_hit < TOL) {
            // If t_near is negative, the ray originates inside the tesseract.
            // The first valid intersection is at the exit point, t_far.
            t_hit = t_far;
            hit_axis = far_axis;
            starts_inside = true;
        }

        // Another object is closer than this one.
        if (t_hit >= res.t) {
            return false;
        }

        // --- We have a valid, closer hit! Populate the hit_result ---
        res.t = t_hit;
        res.p = r.pos + res.t * r.dir; // Hit point in world space
        res.wo = -r.dir;
        res.target = this;

        // Step 5: Calculate the normal vector in local space.
        vec4 local_normal(0.f, 0.f, 0.f, 0.f);
        if (hit_axis != -1) {
            // The normal is an axis vector. The sign depends on whether we hit the
            // -1 or +1 face. If the ray starts inside, the normal points outwards.
            // Otherwise, it points towards the ray's origin.
            float sign = starts_inside ? 
                copysignf(1.0f, local_r_pos.get(hit_axis) + local_r_dir.get(hit_axis) * t_hit) :
                -copysignf(1.0f, local_r_dir.get(hit_axis));
            local_normal.set(hit_axis, sign);
        }

        // Step 6: Transform the normal back to world space and set shading basis.
        res.normal = vec4::normalize(basis.local_to_world(local_normal));
        res.m = transform::get_shading_transform(res.normal);

        return true;
    }

    __host__ __device__ float surface_volume() const override {
        int len = half_len * 2;
		return 8 * len * len * len;
	}

    __host__ __device__ aabb get_bbox_local() const override {
		point4 p_min = point4(-half_len, -half_len, -half_len, -half_len);
		point4 p_max = point4(half_len, half_len, half_len, half_len);
		return aabb(p_min, p_max);
	}

	__host__ __device__ size_t size() const override {
		return sizeof(ncube);
	}

	__device__ virtual void print_gpu() const override {
		printf("[GPU] Hypercube | Corner (%.3f, %.3f, %.3f, %.3f)\n",
		corner.x, corner.y, corner.z, corner.w);
	}
};

#endif