#ifndef CYLINDER_H
#define CYLINDER_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct cylinder : public shape {
	point4 start0, end0;
	point4 start, end;
	bool should_project;
	float radius;

	__host__ __device__ cylinder(): shape{}, start0{0.f, 0.f, 0.f, 0.f}, end0{0.f, 0.f, 0.f, 0.f}, should_project{true}, radius{0.02f} {}

	__host__ __device__ float sdf(const point4& p) const override {
		//point4 pp = basis.world_to_local(p);
		point4 aa = start, bb = end;

		// projection should happen after affine transforms
		if (should_project) {
			aa = aa / (1.f + aa.w);
			bb = bb / (1.f + bb.w);
			aa.w = 0.f;
			bb.w = 0.f;
		}

		vec4  ba = bb - aa;
		vec4  pa = p - aa;
		float baba = vec4::dot(ba, ba);
		float paba = vec4::dot(pa, ba);
		float x = vec4::length(pa * baba - ba * paba) - radius * baba;
		float y = fabsf(paba - baba * 0.5f) - baba * 0.5f;
		float x2 = x * x;
		float y2 = y * y * baba;
		float d = (fmaxf(x, y) < 0.f) ? -fminf(x2, y2) : (((x > 0.f) ? x2 : 0.f) + ((y > 0.f) ? y2 : 0.f));
		float sign_d = (d > 0.f) - (d < 0.f);

		return sign_d * sqrtf(fabsf(d)) / baba;
	}

	__host__ __device__ virtual void rotate(float angle, uint a, uint b) override {
		shape::rotate(angle, a, b);
		start = basis.local_to_world(start0);
		end = basis.local_to_world(end0);
	}
	
	__host__ __device__ virtual void translate(const vec4& t) override {
		shape::translate(t);
		start = basis.local_to_world(start0);
		end = basis.local_to_world(end0);
	}

	__device__ virtual void print_gpu() const override {
		printf("[GPU] Cylinder | Start (%.3f, %.3f, %.3f, %.3f) | End (%.3f, %.3f, %.3f, %.3f) | Radius %.3f | Project %s\n",
		start.x, start.y, start.z, start.w, end.x, end.y, end.z, end.w, radius, should_project ? "true" : "false");
	}
};

#endif // !CYLINDER_H
