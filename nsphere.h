#ifndef NSPHERE_H
#define NSPHERE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct nsphere : public shape {
	point4 center;
	float radius;

	__host__ __device__ nsphere() : shape{}, center { 0.f, 0.f, 0.f, 0.f }, radius{ 0.25f } {}

	__host__ __device__ float sdf(const point4& p) const {
		return (center - p).length() - radius;
	}

	__device__ virtual void print_gpu() const override {
		printf("[GPU] Hypersphere | Center (%.3f, %.3f, %.3f, %.3f) | Radius %.3f\n",
		center.x, center.y, center.z, center.w, radius);
	}
};

#endif