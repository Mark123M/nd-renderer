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

	__host__ __device__ static vec4 reflect(const vec4& wo, const vec4& n) {
		return -wo + 2 * vec4::dot(wo, n) * n;
	}


	__host__ __device__ static float dot(const vec4& a, const vec4& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	// Cross product of the first three elements (w must be equal)
	__host__ __device__ static vec4 cross(const vec4& a, const vec4& b) {
		assert(approx_equals(a.w, b.w));
		//if (!approx_equals(a.w, b.w)) {
		//	printf("[GPU] Invalid cross product %.3f, %.3f", a.w, b.w);
		//}

		return vec4(
			a.y * b.z - a.z * b.y,  // New X component
			a.z * b.x - a.x * b.z,  // New Y component
			a.x * b.y - a.y * b.x,  // New Z component
			0.f
    	);
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

	__host__ __device__ static vec4 rand_unit_disk_vector(uint64_t& pcg_state) {
		while (true) {
			auto p = vec4{ randf_pcg32(-1, 1, pcg_state), 0.f, randf_pcg32(-1, 1, pcg_state), 0.f};
			if (p.length_squared() < 1)
				return p;
		}
	}

	__host__ __device__ static vec4 rand_halfsphere_vector_cosine_weighted(uint64_t& pcg_state) {
		// We need to generate a random point uniformly inside a 3-ball (a sphere volume in 3D)
		// and then project it onto the surface of the half 3-sphere.

		// 1. Generate three uniform random numbers
		float u1 = randf_pcg32(0.f, 1.f, pcg_state);
		float u2 = randf_pcg32(0.f, 1.f, pcg_state);
		float u3 = randf_pcg32(0.f, 1.f, pcg_state);

		// 2. Generate a random direction on the surface of a 2-sphere (standard 3D sphere point picking)
		float phi = 2.f * pi * u1;
		float cos_theta = 1.f - 2.f * u2;
		float sin_theta = sqrtf(fmaxf(0.f, 1.f - cos_theta * cos_theta));

		float dir_x = sin_theta * cosf(phi);
		float dir_z = sin_theta * sinf(phi);
		float dir_w = cos_theta; // Let's use x, z, w for the 3-ball components for clarity

		// 3. Generate a random radius. For a uniform distribution within a 3-ball,
		// the radius 'r' must be sampled such that r^3 is uniform. So, r = cbrt(u3).
		float r = cbrtf(u3);

		// 4. The point inside the 3-ball has coordinates (x, z, w)
		float x = r * dir_x;
		float z = r * dir_z;
		float w = r * dir_w;

		// 5. Project this point onto the half 3-sphere. The "up" direction is y.
		// The squared distance from the center in the "3-ball plane" is r^2.
		float y = sqrtf(fmaxf(0.f, 1.f - r * r));

		// The final cosine-weighted 4D vector
		return vec4(x, y, z, w);
	}

	__host__ __device__ static float cos2_theta(const vec4& w) {
		return w.y * w.y;
	}

	__host__ __device__ static float sin2_theta(const vec4& w) {
		return fmaxf(0.f, 1.f - cos2_theta(w));
	}

	__host__ __device__ static float tan2_theta(const vec4& w) {
		return sin2_theta(w) / cos2_theta(w);
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