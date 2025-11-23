#ifndef PROJECTED_CYLINDER_H
#define PROJECTED_CYLINDER_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

struct projected_cylinder : public shape {
	point4 start0, end0;
	point4 start, end;
	float radius;

	__host__ __device__ projected_cylinder(): shape{}, start0{0.f, 0.f, 0.f, 0.f}, end0{0.f, 0.f, 0.f, 0.f}, radius{0.02f} {}

	__host__ __device__ float sdf(const point4& p) const override {
		point4 aa = start, bb = end;
        aa = aa / (1.f + aa.w);
        bb = bb / (1.f + bb.w);
        aa.w = 0.f;
        bb.w = 0.f;

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

    std::unique_ptr<shape> clone() const override {
        return std::make_unique<projected_cylinder>(*this);
    }

	__host__ __device__ size_t size() const override {
		return sizeof(projected_cylinder);
	}

	__device__ virtual void print_gpu() const override {
		printf("[GPU] Projected Cylinder | Start (%.3f, %.3f, %.3f, %.3f) | End (%.3f, %.3f, %.3f, %.3f) | Radius %.3f\n",
		start0.x, start0.y, start0.z, start0.w, end0.x, end0.y, end0.z, end0.w, radius);
	}
};

#endif // !CYLINDER_H
