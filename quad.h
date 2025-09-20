#ifndef QUAD_H
#define QUAD_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

// Represents a quadrilateral (specifically, a parallelogram) shape.
// It is defined by a corner point `q`, and two edge vectors, `u` and `v`.
struct quad : public shape {
	point4 q; // The corner point of the quad
	vec4 u;   // The first edge vector, from q
	vec4 v;   // The second edge vector, from q

	// Pre-computed values for the intersection test
	vec4 normal;
	float dot_uu, dot_vv, dot_uv;
	float inv_denom;

	// Default constructor creates a degenerate quad at the origin.
	__host__ __device__ quad() 
		: shape{}, q{}, u{}, v{}, normal{}, dot_uu(0), dot_vv(0), dot_uv(0), inv_denom(0) {}

	// Constructor to define the quad and pre-compute necessary values.
	__host__ __device__ quad(const point4& _q, const vec4& _u, const vec4& _v)
		: shape{}, q{_q}, u{_u}, v{_v} {
		
		// Pre-compute the normal vector. The length of the cross product
		// is also the area of the parallelogram.
		vec4 n = vec4::cross(u, v);
		normal = vec4::normalize(n);

		// Pre-compute dot products for the inside/outside test.
		dot_uu = vec4::dot(u, u);
		dot_vv = vec4::dot(v, v);
		dot_uv = vec4::dot(u, v);

		// Pre-compute the inverse of the determinant for solving the
		// linear system in the inside/outside test.
		float denom = dot_uu * dot_vv - dot_uv * dot_uv;
		inv_denom = (denom == 0.f) ? 0.f : 1.f / denom;
	}

	/**
	 * @brief Intersects a ray with the quad.
	 * 
	 * The process is as follows:
	 * 1. Find the intersection of the ray with the infinite plane containing the quad.
	 * 2. If the intersection is valid (in front of ray, closer than other hits),
	 *    check if the hit point lies within the parallelogram bounds.
	 * 3. The inside/outside test is done by projecting the vector from the quad's
	 *    corner (q) to the hit point onto the quad's edge vectors (u and v).
	 */
	__host__ __device__ bool intersect(const ray& r, hit_result& res) const override {
		// A zero inv_denom means the quad is degenerate (u and v are collinear).
		if (inv_denom == 0.f) {
			return false;
		}

		// Step 1: Ray-Plane Intersection
		float denominator = vec4::dot(r.dir, normal);

		// Check if the ray is parallel to the quad's plane.
		if (fabsf(denominator) < TOL) {
			return false;
		}

		// Calculate distance 't' from ray origin to the plane.
		vec4 q_minus_ro = q - r.pos;
		float t = vec4::dot(q_minus_ro, normal) / denominator;

		// Check if the intersection is valid:
		// 1. It must be in front of the ray (t > TOL).
		// 2. It must be closer than any previously found intersection (t < res.t).
		if (t < TOL || t >= res.t) {
			return false;
		}

		// Step 2: Inside/Outside Test
		// Find the hit point on the plane.
		point4 p = r.pos + t * r.dir;
		vec4 w = p - q;

		// Solve the linear system w = alpha*u + beta*v for alpha and beta.
		// These are the barycentric-like coordinates of the hit point within the quad.
		float dot_wv = vec4::dot(w, v);
		float dot_wu = vec4::dot(w, u);
		
		float alpha = (dot_vv * dot_wu - dot_uv * dot_wv) * inv_denom;
		float beta  = (dot_uu * dot_wv - dot_uv * dot_wu) * inv_denom;

		// If alpha or beta are outside the [0, 1] range, the hit point
		// is on the plane but outside the quad's boundaries.
		if (alpha < 0.f || alpha > 1.f || beta < 0.f || beta > 1.f) {
			return false;
		}

		// Step 3: A valid, closer intersection was found. Populate the hit_result struct.
		res.t = t;
		res.p = p;                 // Hit point in world space
		res.wo = -r.dir;           // Negative ray direction in world space
		res.target = this;         // Pointer to this shape
		
		// The normal is pre-calculated. We ensure it faces the incoming ray.
		// This is a common convention for single-sided lighting.
		// Your hit_result comment says "always pointing outwards", which can be
		// interpreted as against the incoming ray, so we'll implement that.
		res.normal = vec4::dot(r.dir, normal) < 0 ? normal : -normal;
		
		// Create the shading basis transform
		res.m = transform::get_shading_transform(res.normal);

		return true;
	}

	__host__ __device__ bool sample(shape_sample& ss, const hit_result& res, uint64_t& pcg_state) const override {
		float ru = randf_pcg32(0.f, 1.f, pcg_state), rv = randf_pcg32(0.f, 1.f, pcg_state);
		point4 p = q + ru * u + rv * v;
		ss.p = p;

		float area = vec4::cross(u, v).length();
		vec4 wi = vec4::normalize(p - res.p);
		float len = vec4::length(p - res.p);
		float jacobian = fabsf(vec4::dot(normal, -wi)) / (len * len * len);
		ss.pdf = (1 / area) / jacobian;
	}

    // Returns the size of the object in memory.
	__host__ __device__ size_t size() const override {
		return sizeof(quad);
	}
	
    // Prints information about the quad from the GPU.
	__device__ virtual void print_gpu() const override {
		printf("[GPU] Quad | Corner (%.3f, %.3f, %.3f, %.3f) | U (%.3f, %.3f, %.3f) | V (%.3f, %.3f, %.3f)\n",
		q.x, q.y, q.z, q.w, u.x, u.y, u.z, v.x, v.y, v.z);
	}
};

#endif // QUAD_H
