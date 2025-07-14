#ifndef NSPHERE_H
#define NSPHERE_H

#include "vec4.h"
#include "color.h"


struct nsphere {
	point4 center;
	float radius;
	color albedo;
	int n;

	nsphere(const point4& center, float radius, int n = 4) : center{ center }, radius{ radius }, albedo{ 1.f, 1.f, 1.f }, n { n } {}
	nsphere(const point4& center, float radius, const color& albedo0, int n = 4) : center{ center }, radius{ radius }, albedo{ albedo0 }, n{ n } {}

	float sdf(const point4& p) {
		return (center - p).length() - radius;
	}
};

#endif