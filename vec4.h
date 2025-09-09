#ifndef VEC4_H
#define VEC4_H

#include <cmath>
#include <iostream>
#include "util.h"

struct vec4;
__host__ __device__ vec4 operator*(float k, const vec4& v);

struct vec4 {
	float x, y, z, w;

	__host__ __device__ vec4 operator-() const { return { -x, -y, -z, -w }; }

	__host__ __device__ vec4 operator+(const vec4& v) const { return { x + v.x, y + v.y, z + v.z, w + v.w }; }

	__host__ __device__ vec4 operator-(const vec4& v) const { return { x - v.x, y - v.y, z - v.z, w - v.w }; }

	__host__ __device__ vec4 operator*(float k) const { return { k * x, k * y, k * z, k * w }; }

	__host__ __device__ vec4 operator/(float k) const {
		assert(k != 0.f);
		return this->operator*(1.f / k);
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

	__host__ __device__ vec4& operator*=(float k) {
		x *= k;
		y *= k;
		z *= k;
		w *= k;
		return *this;
	}

	__host__ __device__ vec4& operator/=(float k) {
		assert(k != 0.f);
		return this->operator*=(1.f / k);
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

	__host__ __device__ static vec4 reflect(const vec4& wo) {
		return vec4(-wo.x, wo.y, -wo.z, -wo.w);
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

	// random vector in unit 3-sphere (naive)
	__host__ __device__ static vec4 rand_unit_vector(uint64_t& pcg_state) {
		while (true) {
			vec4 p = vec4{ randf_pcg32(-1, 1, pcg_state), randf_pcg32(-1, 1, pcg_state), randf_pcg32(-1, 1, pcg_state), randf_pcg32(-1, 1, pcg_state)};
			float lensq = p.length_squared();
			if (EPSILON < lensq && lensq <= 1) {
				return p / sqrtf(lensq);
			}
		}
	}

	// random vector in unit 3-half-sphere (naive)
	__host__ __device__ static vec4 rand_halfsphere_vector(uint64_t& pcg_state) {
		vec4 unit = rand_unit_vector(pcg_state);

		if (unit.y > 0) {
			return unit;
		} else {
			return -unit;
		}
	}

	__host__ __device__ void print() {
		printf("vec4(%.3f, %.3f, %.3f, %.3f)\n", x, y, z, w);
	}
};

__host__ __device__ vec4 operator*(float k, const vec4& v) { return v * k; }

// host constants
constexpr vec4 delta_x{EPSILON, 0.f, 0.f, 0.f};
__constant__ vec4 d_delta_x;
constexpr vec4 delta_y{0.f, EPSILON, 0.f, 0.f};
__constant__ vec4 d_delta_y;
constexpr vec4 delta_z{0.f, 0.f, EPSILON, 0.f};
__constant__ vec4 d_delta_z;
constexpr vec4 delta_w{0.f, 0.f, 0.f, EPSILON};
__constant__ vec4 d_delta_w;

constexpr vec4 standard_x{1.f, 0.f, 0.f, 0.f};
constexpr vec4 standard_y{0.f, 1.f, 0.f, 0.f};
constexpr vec4 standard_z{0.f, 0.f, 1.f, 0.f};
constexpr vec4 standard_w{0.f, 0.f, 0.f, 1.f};

struct point4 {
	float x, y, z, w;
	
	__host__ __device__ point4 operator+(const vec4& v) const { return { x + v.x, y + v.y, z + v.z, w + v.w }; }

	__host__ __device__ point4 operator-(const vec4& v) const { return { x - v.x, y - v.y, z - v.z, w - v.w }; }

	__host__ __device__ vec4 operator-(const point4& p) const { return { x - p.x, y - p.y, z - p.z, w - p.w }; }

	__host__ __device__ point4 operator/(float k) const {
		assert(k != 0.f);
		return { x / k, y / k, z / k, w / k };
	}

	__host__ __device__ point4& operator+=(const vec4& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return *this;
	}

	__host__ __device__ point4& operator-=(const vec4& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
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
};

__managed__ point4 origin{0.f, 0.f, 0.f, 0.f};

#endif