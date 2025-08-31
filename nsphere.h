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
		float c = vec4::dot(oc, oc) - radius * radius; // r^2 = 1*1 = 1 for a unit sphere

		// Step 3: Solve for t using the quadratic formula.
		// First, check the discriminant to see if there are any real solutions.
		float discriminant = b * b - 4.f * a * c;

		if (discriminant < 0.f) {
			// No real roots, so no intersection.
			return false;
		}

		// Calculate the two potential solutions for t.
		float sqrt_disc = sqrtf(discriminant);
		float t0 = (-b - sqrt_disc) / (2.f * a);
		float t1 = (-b + sqrt_disc) / (2.f * a);
		
		// Step 4: Find the closest valid intersection point in front of the ray.
		// A valid t must be greater than a small epsilon (T_MIN) to avoid self-intersection.
		float t;
		if (t0 > TOL) {
			t = t0;
		} else if (t1 > TOL) {
			t = t1;
		} else {
			// Both intersections are behind the ray's origin or too close.
			return false;
		}

		// Another object is closer
		if (t >= res.t) {
			return false;
		}

		// Step 5: An intersection was found. Populate the hit_result struct.
		res.t = t;
		
		// The hit point in world space.
		res.p = r.pos + t * r.dir;

		// The negative direction of the incoming ray.
		res.wo = -r.dir;

		// Calculate the normal vector.
		// 1. Find the hit point in local space.
		point4 local_p = local_r_pos + t * local_r_dir;
		// 2. For a sphere centered at the origin, the normal is just the normalized position vector.
		vec4 local_normal = vec4::normalize(local_p - origin);
		// 3. Transform the normal back to world space. Normals are transformed by the
		//    inverse transpose of the original transformation matrix.
		res.normal = basis.local_to_world(local_normal);

		// Store a pointer to the shape that was hit.
		res.target = this;
		res.m = transform::get_shading_transform(res.normal);

		return true;
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