#ifndef NSPHERE_H
#define NSPHERE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct nsphere : public shape {
	float radius;

	__host__ __device__ nsphere() : shape{}, radius{ 0.25f } {}

	__host__ __device__ float sdf(const point4& p) const {
		point4 pp = basis.world_to_local(p);
		return (origin - pp).length() - radius;
	}

	__host__ __device__ bool intersect(const ray& r, hit_result& res) const override {
		point4 local_r_pos = basis.world_to_local(r.pos); // are these too expensive?
		vec4 local_r_dir = basis.world_to_local(r.dir);
		
		vec4 oc = local_r_pos - origin;
		float a = vec4::dot(local_r_dir, local_r_dir);
		float b = 2.f * vec4::dot(local_r_dir, oc);
		float c = vec4::dot(oc, oc) - radius * radius;
		float discriminant = b * b - 4.f * a * c;

		if (discriminant < 0.f) {
			return false;
		}

		// calculate the two potential solutions for t
		float sqrt_disc = sqrtf(discriminant);
		float t0 = (-b - sqrt_disc) / (2.f * a);
		float t1 = (-b + sqrt_disc) / (2.f * a);
		float t;

		if (t0 > TOL) {
			t = t0;
		} else if (t1 > TOL) {
			t = t1;
		} else {
			// both intersections are behind the ray's origin or too close
			return false;
		}

		if (t >= res.t) {
			return false;
		}

		// the hit point in local space
		point4 local_p = local_r_pos + t * local_r_dir;
		// the local normal is just the normalized position vector
		vec4 local_normal = vec4::normalize(local_p - origin);

		res.t = t;
		res.p = r.pos + t * r.dir;
		res.wo = -r.dir;
		res.normal = basis.local_to_world(local_normal);
		res.target = this;
		res.m = transform::get_shading_transform(res.normal);
		return true;
	}

	__host__ __device__ float surface_volume() const override {
		return 2 * pi * pi * radius * radius * radius;
	}

	__host__ __device__ bool sample(shape_sample& ss, const hit_result& res, uint64_t& pcg_state) const override {
		point4 local_p = origin + radius * vec4::rand_unit_vector(pcg_state);
		ss.p = basis.local_to_world(local_p);
		ss.pdf = 1.f / surface_volume();
		return true;
	}

	__host__ __device__ aabb get_bbox_local() const override {
		point4 p_min = point4(-radius, -radius, -radius, -radius);
		point4 p_max = point4(radius, radius, radius, radius);
		return aabb(p_min, p_max);
	}

	std::unique_ptr<shape> clone() const override {
        return std::make_unique<nsphere>(*this);
    }

	__host__ __device__ size_t size() const override {
		return sizeof(nsphere);
	}
	
	__device__ virtual void print_gpu() const override {
		point4 center = basis.get_pos();
		printf("[GPU] Hypersphere | Center (%.3f, %.3f, %.3f, %.3f) | Radius %.3f\n",
		center.x, center.y, center.z, center.w, radius);
	}
};

#endif