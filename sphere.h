#ifndef SPHERE_H
#define SPHERE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct sphere : public shape {
	float radius;
	float half_w;

	__host__ __device__ sphere() : shape{}, radius{ 0.25f }, half_w{ TOL } {}

	__host__ __device__ float sdf(const point4& p) const {
		point4 pp = basis.world_to_local(p);
		return (origin - pp).length() - radius;
	}

	__host__ __device__ bool intersect(const ray& r, hit_result& res) const override {
    	// --- 1. Transform ray to object's local space ---
		point4 local_r_pos = basis.world_to_local(r.pos);
		vec4 local_r_dir = basis.world_to_local(r.dir);
		vec4 oc = local_r_pos - origin;

		// --- 2. Intersect with the infinite 3D "spherinder" ---
		// This finds the interval [t_sphere_enter, t_sphere_exit]
		float a = vec4::dot3(local_r_dir, local_r_dir);
		float b = 2.f * vec4::dot3(local_r_dir, oc);
		float c = vec4::dot3(oc, oc) - radius * radius;
		float discriminant = b * b - 4.f * a * c;

		if (discriminant < 0.f) {
			// Ray misses the infinite spherinder entirely
			return false;
		}

		float sqrt_disc = sqrtf(discriminant);
		float t_sphere_enter = (-b - sqrt_disc) / (2.f * a);
		float t_sphere_exit = (-b + sqrt_disc) / (2.f * a);

		// --- 3. Intersect with the infinite W-slab ---
		// This finds the interval [t_slab_enter, t_slab_exit]
		float t_slab_enter, t_slab_exit;
		
		// Check if ray is parallel to the slab planes
		if (fabsf(local_r_dir.w) < TOL) {
			// If parallel, check if the ray origin is inside the slab's w-range
			if (fabsf(oc.w) > half_w) {
				return false; // Parallel and outside, will never hit
			}
			// Parallel and inside, the slab imposes no t-limit
			t_slab_enter = -FLT_MAX;
			t_slab_exit = FLT_MAX;
		} else {
			// Ray is not parallel, calculate intersections with the two w-planes
			float t_w0 = (-half_w - oc.w) / local_r_dir.w;
			float t_w1 = (half_w - oc.w) / local_r_dir.w;
			t_slab_enter = fminf(t_w0, t_w1);
			t_slab_exit = fmaxf(t_w0, t_w1);
		}
		
		// --- 4. Find the intersection of the two intervals ---
		// The final hit interval is [t_near, t_far]
		float t_near = fmaxf(t_sphere_enter, t_slab_enter);
		float t_far = fminf(t_sphere_exit, t_slab_exit);

		if (t_near >= t_far) {
			// The intervals do not overlap, so no hit
			return false;
		}

		// --- 5. Find the first valid intersection time 't' ---
		float t;
		if (t_near > TOL) {
			t = t_near;
		} else if (t_far > TOL) {
			t = t_far; // Ray origin is inside the object, we hit the back face
		} else {
			// Both intersections are behind the ray's origin or too close
			return false;
		}

		if (t >= res.t) {
			// A closer intersection has already been found
			return false;
		}
		
		// --- 6. Calculate hit point and normal IN LOCAL SPACE ---
		point4 local_p = local_r_pos + t * local_r_dir; // Note: oc is local_r_pos - origin
		vec4 local_normal;

		// To determine the normal, we must know which surface was hit.
		// This is determined by which interval's start defined t_near.
		// A small tolerance is needed for floating point comparison.
		if (t_near > t_sphere_enter + TOL) {
			// Hit a flat cap because the slab intersection was the limiting factor.
			// The normal is purely in the W direction. Its sign is opposite to the
			// ray's w-direction, ensuring it points "out" of the object.
			local_normal = vec4(0.f, 0.f, 0.f, -copysignf(1.f, local_r_dir.w));
		} else {
			// Hit the curved "side" of the spherinder.
			// The normal has no W component.
			local_normal = vec4(local_p.x, local_p.y, local_p.z, 0.f);
		}

		// --- 7. Finalize hit result in world space ---
		res.t = t;
		res.p = r.pos + t * r.dir;
		res.wo = -r.dir;
		res.normal = vec4::normalize(basis.local_to_world(local_normal)); // Normalize after transform
		res.target = this;
		res.m = transform::get_shading_transform(res.normal);

		return true;
	}

	__host__ __device__ virtual float surface_volume() const {
		return 4 * pi * radius * radius;
	}

	__host__ __device__ bool sample(shape_sample& ss, const hit_result& res, uint64_t& pcg_state) const override {
		ss.p = origin + vec4::rand_unit_vector3(pcg_state) * radius;
		ss.pdf = 1.f / surface_volume();
		return true;
	}

	__host__ __device__ aabb get_bbox_local() const override {
		point4 p_min = point4(-radius, -radius, -radius, -half_w);
		point4 p_max = point4(radius, radius, radius, half_w);
		return aabb(p_min, p_max);
	}

	std::unique_ptr<shape> clone() const override {
        return std::make_unique<sphere>(*this);
    }

	__host__ __device__ size_t size() const override {
		return sizeof(sphere);
	}
	
	__device__ virtual void print_gpu() const override {
		point4 center = basis.get_pos();
		printf("[GPU] Sphere | Center (%.3f, %.3f, %.3f, %.3f) | Radius %.3f\n",
		center.x, center.y, center.z, center.w, radius);
	}
};

#endif