#ifndef VEC4_H
#define VEC4_H

#include <cmath>
#include <iostream>
#include "util.h"

struct vec4;
__host__ __device__ vec4 operator*(float k, const vec4& v);

using point4 = vec4;

struct vec4 {
	float x, y, z, w;

	__host__ __device__ vec4() {}

	__host__ __device__ vec4(float x0, float y0, float z0, float w0) : x{ x0 }, y{ y0 }, z{ z0 }, w{ w0 } {}

	__host__ __device__ vec4 operator-() const { return { -x, -y, -z, -w }; }

	__host__ __device__ vec4 operator+(const vec4& v) const { return { x + v.x, y + v.y, z + v.z, w + v.w }; }

	__host__ __device__ vec4 operator-(const vec4& v) const { return { x - v.x, y - v.y, z - v.z, w - v.w }; }

	__host__ __device__ vec4 operator*(const vec4& v) const { return { x * v.x, y * v.y, z * v.z, w * v.w }; }

	__host__ __device__ vec4 operator/(float k) const {
		assert(k != 0.f);
		return (1 / k) * (*this);
	}

	__host__ __device__ vec4& operator+=(const vec4& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return *this;
	}

	__host__ __device__ vec4& operator-=(const vec4& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
		return *this;
	}

	__host__ __device__ vec4& operator *=(const vec4& v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		w *= v.w;
		return *this;
	}

	__host__ __device__ vec4& operator*=(float k) {
		x *= k;
		y *= k;
		z *= k;
		w *= k;
		return *this;
	}

	__host__ __device__ vec4& operator/=(float k) {
		assert(k != 0.f);
		x /= k;
		y /= k;
		z /= k;
		w /= k;
		return *this;
	}

	__host__ __device__ float get(uint idx) const {
		switch (idx) {
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		case 3:
			return w;
		}

		return 0.f;
	}

	__host__ __device__ void set(uint idx, float val) {
		switch (idx) {
		case 0:
			x = val;
			break;
		case 1:
			y = val;
			break;
		case 2:
			z = val;
			break;
		case 3:
			w = val;
			break;
		}
	}

	__host__ __device__ float length() const {
		return sqrtf(dot(*this, *this));
	}

	__host__ __device__ float length_squared() const {
		return dot(*this, *this);
	}

	__host__ __device__ bool near_zero() const {
		return fabsf(x) < TOL && fabsf(y) < TOL && fabsf(z) < TOL; // Use fabsf
	}

	__host__ __device__ static vec4 normalize(const vec4& v) {
		return v / v.length();
	}

	__host__ __device__ static vec4 reflect(const vec4& v, const vec4& n) {
		return v - 2.f * dot(v, n) * n;
	}

	__host__ __device__ static vec4 refract(const vec4& v, const vec4& n, float eta) {
		float cos_theta = fminf(dot(-v, n), 1.f);
		vec4 perp = eta * (v + cos_theta * n);
		vec4 parallel = -sqrtf(fabsf(1.f - perp.length_squared())) * n;
		return perp + parallel;
	}

	__host__ __device__ static float dot(const vec4& a, const vec4& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	__host__ __device__ static float max_comp(const vec4& v) {
		return fmaxf(v.x, fmaxf(v.y, fmaxf(v.z, v.w)));
	}

	__host__ __device__ static vec4 max(const vec4& v, float k) {
		return { fmaxf(v.x, k), fmaxf(v.y, k), fmaxf(v.z, k), fmaxf(v.w, k) };
	}

	__host__ __device__ static vec4 abs(const vec4& v) {
		return { fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w) };
	}

	__host__ __device__ static float length(const vec4& v) {
		return v.length();
	}
};

__host__ __device__ vec4 operator*(float k, const vec4& v) { return { k * v.x, k * v.y, k * v.z, k * v.w }; }

__host__ __device__ vec4 operator*(const vec4& v, float k) { return k * v; }

// host constants
const vec4 delta_x(EPSILON, 0.f, 0.f, 0.f);
__constant__ vec4 d_delta_x;
const vec4 delta_y(0.f, EPSILON, 0.f, 0.f);
__constant__ vec4 d_delta_y;
const vec4 delta_z(0.f, 0.f, EPSILON, 0.f);
__constant__ vec4 d_delta_z;
const vec4 delta_w(0.f, 0.f, 0.f, EPSILON);
__constant__ vec4 d_delta_w;

#endif