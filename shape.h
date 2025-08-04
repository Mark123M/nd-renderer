#ifndef SHAPE_H
#define SHAPE_H

#include "vec4.h"
#include "color.h"
#include "transform.h"

struct shape {
	transform basis;
	color albedo;

	__host__ __device__ shape() : basis{}, albedo{ 1.f, 1.f, 1.f } {}

	__host__ __device__ virtual float sdf(const point4& p) const = 0;

	__device__ virtual void print_gpu() const = 0;
};

#endif