#ifndef CYLINDER_H
#define CYLINDER_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct cylinder : public shape {
	point4 a;
	point4 b;
	bool should_project;
	float radius;

	cylinder(const point4& start, const point4& end, bool should_project, float radius = 0.02f, int n = 4):
		shape{ n }, a{ start }, b{ end }, should_project{ should_project }, radius {radius} {}
	cylinder(const point4& start, const point4& end, bool should_project, const color& albedo, float radius = 0.02f, int n = 4):
		shape{ albedo, n }, a{ start }, b{ end }, should_project{ should_project }, radius {radius} {}

	float sdf(const point4& p) const override {
		//point4 pp = transform.world_to_local(p);

		point4 aa = a;
		point4 bb = b;

		// projection should happen after affine transforms
		if (should_project) {
			aa = aa / (1.f + aa.w);
			bb = bb / (1.f + bb.w);
			aa.w = 0;
			bb.w = 0;
		}

		vec4  ba = bb - aa;
		vec4  pa = p - aa;

		float baba = dot(ba, ba);
		float paba = dot(pa, ba);
		float x = length(pa * baba - ba * paba) - radius * baba;
		float y = abs(paba - baba * 0.5) - baba * 0.5;
		float x2 = x * x;
		float y2 = y * y * baba;
		float d = (std::max(x, y) < 0.0) ? -std::min(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
		float sign_d = (d > 0) - (d < 0);

		return sign_d * sqrt(abs(d)) / baba;
	}
};

#endif // !CYLINDER_H
