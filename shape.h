#ifndef SHAPE_H
#define SHAPE_H

#include "vec4.h"
#include "color.h"
#include "transform.h"
#include "material.h"

enum shape_type {
    CYLINDER, PROJECTED_CYLINDER, HYPERSPHERE, HYPERCUBE
};

struct shape;

struct hit_result {
	point4 p;
	vec4 wo; // negative direction of ray
	vec4 normal; // always pointing outwards
	transform m;
	float t = MAX_RAY_DIST;
	const shape* target;
};

struct shape_sample {
	point4 p;
	float pdf = 1.f;
};

struct shape {
	transform basis;
	size_t mat_idx;

	__host__ __device__ shape() : basis{identity_affine, identity_affine} {}

	__host__ __device__ virtual float sdf(const point4& p) const {
		return 0.f;
	}

	__host__ __device__ virtual bool intersect(const ray& r, hit_result& res) const {
		return false;
	}

	__host__ __device__ virtual bool sample(shape_sample& ss, const hit_result& res, uint64_t& pcg_state) const {
		return false;
	}

	__host__ __device__ virtual void rotate(float angle, uint a, uint b) {
		basis.rotate(angle, a, b);
	}
	
	__host__ __device__ virtual void translate(const vec4& t) {
		basis.translate(t);
	}

	__host__ __device__ void rotate_xy(float angle) {
		rotate(angle, 0, 1);
	}

	__host__ __device__ void rotate_xz(float angle) {
		rotate(angle, 0, 2);
	}

	__host__ __device__ void rotate_xw(float angle) {
		rotate(angle, 0, 3);
	}

	__host__ __device__ void rotate_yz(float angle) {
		rotate(angle, 1, 2);
	}

	__host__ __device__ void rotate_yw(float angle) {
		rotate(angle, 1, 3);
	}

	__host__ __device__ void rotate_zw(float angle) {
		rotate(angle, 2, 3);
	}

	__host__ __device__ virtual size_t size() const = 0;

	__device__ virtual void print_gpu() const = 0;
};

#endif