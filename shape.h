#ifndef SHAPE_H
#define SHAPE_H

#include "vec4.h"
#include "color.h"
#include "transform.h"

struct shape {
	color albedo;
	int n;
	transform transform;

	shape(int n) : albedo{ 1.f, 1.f, 1.f }, n { n } {}
	shape(const color& albedo, int n) : albedo{albedo}, n{n} {}

	virtual float sdf(const point4& p) const = 0;
};

#endif