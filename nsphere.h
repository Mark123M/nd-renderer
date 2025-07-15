#ifndef NSPHERE_H
#define NSPHERE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct nsphere : public shape {
	point4 center;
	float radius;

	nsphere(const point4& center, float radius, int n = 4) : shape{ n }, center { center }, radius{ radius } {}
	nsphere(const point4& center, float radius, const color& albedo0, int n = 4) : shape{ albedo0, n }, center { center }, radius{ radius } {}

	float sdf(const point4& p) const {
		return (center - p).length() - radius;
	}
};

#endif