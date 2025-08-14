#ifndef NCUBE_H
#define NCUBE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct ncube : public shape {
	point4 corner;

	__host__ __device__ ncube() : shape{}, corner{ 0.f, 0.f, 0.f, 0.f } {}

	__host__ __device__ float sdf(const point4& p) const override {
		point4 pp = basis.world_to_local(p); //multiply_matrix_vec4(example_matrix, p);
		
		vec4 q = vec4::abs(pp) - corner;
		float d = vec4::max(q, 0.f).length() + fminf(vec4::max_comp(q), 0.f);
		return d;
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