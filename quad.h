#ifndef QUAD_H
#define QUAD_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

// Represents a quadrilateral (specifically, a parallelogram) shape.
// It is defined by a corner point `q`, and two edge vectors, `u` and `v`.
struct quad : public shape {
	point4 o;
	vec4 u, v;
	vec4 normal;
	vec4 w; // Used to find local coordinates
	float D; // Ax + By + Cz = D

	// Constructor to define the quad and pre-compute necessary values.
	__host__ __device__ quad(const point4& _q, const vec4& _u, const vec4& _v)
		: shape{}, o{_q}, u{_u}, v{_v} {
		assert(o.w == 0.f && u.w == 0.f && v.w == 0.f);

		vec4 n = vec4::cross(u, v);
		normal = vec4::normalize(n);
		D = vec4::dot3(normal, o - origin);
		w = n / vec4::dot3(n, n);
	}

	__host__ __device__ bool intersect(const ray& r, hit_result& res) const override {
		float t;
		
		//if (approx_equals(r.pos.w, 0.f)) {
			float denom = vec4::dot3(normal, r.dir);
			// Parallel to plane
			if (approx_equals(denom, 0.f)) {
				return false;
			}

			t = (D - vec4::dot3(normal, r.pos - origin)) / denom;
		/*} else {
			if (approx_equals(r.dir.w, 0.f)) {
				return false;
			}

			t = -r.pos.w / r.dir.w;
		}*/

		if (t < TOL) {
			return false;
		}

		if (t >= res.t) {
			return false;
		}

		point4 p = r.pos + t * r.dir;

		//if (!approx_equals(vec4::dot3(p - o, normal), 0.f)) {
		//	return false;
		//}

		float alpha = vec4::dot3(w, vec4::cross(p - o, v));
		float beta = vec4::dot3(w, vec4::cross(u, p - o));

		if (alpha < 0.f || alpha > 1.f || beta < 0.f || beta > 1.f) {
			return false;
		}

		// Step 3: A valid, closer intersection was found. Populate the hit_result struct.
		res.t = t;
		res.p = p;                 // Hit point in world space
		res.wo = -r.dir;           // Negative ray direction in world space
		res.target = this;         // Pointer to this shape
		res.normal = vec4::dot(r.dir, normal) < 0.f ? normal : -normal;
		res.m = transform::get_shading_transform(res.normal);

		return true;
	}

	__host__ __device__ bool sample(shape_sample& ss, const hit_result& res, uint64_t& pcg_state) const override {
		float ru = randf_pcg32(0.f, 1.f, pcg_state), rv = randf_pcg32(0.f, 1.f, pcg_state);
		point4 p = o + ru * u + rv * v;
		ss.p = p;

		float area = vec4::cross(u, v).length();
		vec4 wi = vec4::normalize(p - res.p);
		float len = vec4::length(p - res.p);
		float jacobian = fabsf(vec4::dot(normal, -wi)) / (len * len * len);
		ss.pdf = (1 / area) / jacobian;

		return true;
	}

    // Returns the size of the object in memory.
	__host__ __device__ size_t size() const override {
		return sizeof(quad);
	}
	
    // Prints information about the quad from the GPU.
	__device__ virtual void print_gpu() const override {
		printf("[GPU] Quad | Corner (%.3f, %.3f, %.3f, %.3f) | U (%.3f, %.3f, %.3f) | V (%.3f, %.3f, %.3f)\n",
		o.x, o.y, o.z, o.w, u.x, u.y, u.z, v.x, v.y, v.z);
	}
};

#endif // QUAD_H
