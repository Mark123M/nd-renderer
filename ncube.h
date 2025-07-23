#ifndef NCUBE_H
#define NCUBE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct ncube : public shape {
	point4 corner;

	ncube(const point4& corner, int n = 4) : shape{n}, corner{ corner } {}
	ncube(const point4& corner, const color& albedo0, int n = 4) : shape{ albedo0, n }, corner{ corner } {}

	float sdf(const point4& p) const override {
		point4 pp = transform.world_to_local(p); //multiply_matrix_vec4(example_matrix, p);
		
		vec4 q = abs(pp) - corner;
		float d = max(q, 0).length() + std::min(max_comp(q), 0.f);
		return d;
	}
};

#endif