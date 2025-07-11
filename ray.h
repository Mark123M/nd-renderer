#ifndef RAY_H
#define RAY_H

#include "vec4.h"

class entity;

using point4 = vec4;

struct ray {
	point4 origin;
	vec4 dir;

	ray() : origin{ 0.f, 0.f, 0.f, 0.f }, dir{ 1.f, 0.f, 0.f, 0.f } {}
	ray(const point4& origin, const vec4& direction) : origin{ origin }, dir{ direction } {}
	ray(const point4& origin, const vec4& direction, const entity* target) : origin{ origin }, dir{ direction } {}

	point3 at(float t) const {
		return origin + t * dir;
	}
};

#endif