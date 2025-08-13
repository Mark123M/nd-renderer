#ifndef SHAPE_H
#define SHAPE_H

#include "vec4.h"
#include "color.h"
#include "transform.h"

enum shape_type {
    NONE, CYLINDER, PROJECTED_CYLINDER, HYPERSPHERE, HYPERCUBE
};

struct shape {
	transform basis;
	color albedo;

	__host__ __device__ shape() : basis{}, albedo{ 1.f, 1.f, 1.f } {}

	__host__ __device__ virtual float sdf(const point4& p) const = 0;

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