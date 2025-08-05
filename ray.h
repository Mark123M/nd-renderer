#ifndef RAY_H
#define RAY_H

#include "vec4.h"
#include "util.h"

class entity;

struct ray {
	point4 pos;
	vec4 dir;

	__host__ __device__ ray() : pos{ 0.f, 0.f, 0.f, 0.f }, dir{ 1.f, 0.f, 0.f, 0.f } {}
	__host__ __device__ ray(const point4& origin, const vec4& direction) : pos{ origin }, dir{ direction } {
		assert(fabsf(direction.length_squared() - 1.f) <= TOL);
	}
	__host__ __device__ ray(const point4& origin, const vec4& direction, const entity* target) : pos{ origin }, dir{ direction } {
		assert(fabsf(direction.length_squared() - 1.f) <= TOL);
	}

	__host__ __device__ void march(float dist) {
		pos += dist * dir;
	}
};

#endif