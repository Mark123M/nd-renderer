#ifndef RAY_H
#define RAY_H

#include "vec4.h"
#include <cassert>
#include "util.h"

class entity;

struct ray {
	point4 pos;
	vec4 dir;

	ray() : pos{ 0.f, 0.f, 0.f, 0.f }, dir{ 1.f, 0.f, 0.f, 0.f } {}
	ray(const point4& origin, const vec4& direction) : pos{ origin }, dir{ direction } {
		assert(std::abs(direction.length_squared() - 1.f) <= TOL);
	}
	ray(const point4& origin, const vec4& direction, const entity* target) : pos{ origin }, dir{ direction } {
		assert(std::abs(direction.length_squared() - 1.f) <= TOL);
	}

	void march(float dist) {
		pos += dist * dir;
	}
};

#endif