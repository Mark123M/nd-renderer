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

	__host__ __device__ cylinder(): shape{}, a{0.f, 0.f, 0.f, 0.f}, b{0.f, 0.f, 0.f, 0.f}, should_project{true}, radius{0.02f} {}

	__host__ __device__ float sdf(const point4& p) const override {
		//point4 pp = basis.world_to_local(p);
		point4 aa = basis.local_to_world(a);
		point4 bb = basis.local_to_world(b);

		// projection should happen after affine transforms
		if (should_project) {
			aa = aa / (1.f + aa.w);
			bb = bb / (1.f + bb.w);
			aa.w = 0;
			bb.w = 0;
		}

		vec4  ba = bb - aa;
		vec4  pa = p - aa;
		float baba = vec4::dot(ba, ba);
		float paba = vec4::dot(pa, ba);
		float x = vec4::length(pa * baba - ba * paba) - radius * baba;
		float y = fabsf(paba - baba * 0.5) - baba * 0.5;
		float x2 = x * x;
		float y2 = y * y * baba;
		float d = (fmaxf(x, y) < 0.0) ? -fminf(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
		float sign_d = (d > 0) - (d < 0);

		return sign_d * sqrtf(fabsf(d)) / baba;
	}

	__device__ virtual void print_gpu() const override {
		printf("[GPU] Cylinder | Start (%.3f, %.3f, %.3f, %.3f) | End (%.3f, %.3f, %.3f, %.3f) | Radius %.3f | Project %s\n",
		a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, radius, should_project ? "true" : "false");
	}
};

#endif // !CYLINDER_H
