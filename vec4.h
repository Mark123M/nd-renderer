#ifndef VEC4_H
#define VEC4_H

#include <cmath>
#include <iostream>
#include <cassert>

class vec4;
inline vec4 operator*(float k, const vec4& v);

using point4 = vec4;

struct vec4 {
	float x, y, z, w;

	vec4() : x{ 0.f }, y{ 0.f }, z{ 0.f }, w{ 0.f } {}

	vec4(float x0, float y0, float z0, float w0) : x{ x0 }, y{ y0 }, z{ z0 }, w{ w0 } {}

	vec4 operator-() const { return { -x, -y, -z, -w }; }

	vec4 operator+(const vec4& v) const { return { x + v.x, y + v.y, z + v.z, w + v.w }; }

	vec4 operator-(const vec4& v) const { return { x - v.x, y - v.y, z - v.z, w - v.w }; }

	vec4 operator*(const vec4& v) const { return { x * v.x, y * v.y, z * v.z, w * v.w }; }

	vec4 operator/(float k) const {
		assert(k != 0);
		return (1 / k) * (*this);
	}

	vec4& operator+=(const vec4& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return *this;
	}

	vec4& operator-=(const vec4& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
		return *this;
	}

	vec4& operator *=(const vec4& v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		w *= v.w;
		return *this;
	}

	vec4& operator*=(float k) {
		x *= k;
		y *= k;
		z *= k;
		w *= k;
		return *this;
	}

	vec4& operator/=(float k) {
		assert(k != 0);
		x /= k;
		y /= k;
		z /= k;
		w /= k;
		return *this;
	}

	float get(int idx) const {
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

	void set(int idx, float val) {
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

	float length() const {
		return std::sqrt(dot(*this, *this));
	}

	float length_squared() const {
		return dot(*this, *this);
	}

	bool near_zero() const {
		float t = 1e-8f;
		return std::fabs(x) < t && std::fabs(y) < t && std::fabs(z) < t;
	}

	inline static vec4 normalize(const vec4& v) {
		return v / v.length();
	}

	inline static vec4 reflect(const vec4& v, const vec4& n) {
		return v - 2 * dot(v, n) * n;
	}

	inline static vec4 refract(const vec4& v, const vec4& n, float eta) {
		float cos_theta = std::fmin(dot(-v, n), 1.f);
		vec4 perp = eta * (v + cos_theta * n);
		vec4 parallel = -std::sqrt(std::fabs(1.f - perp.length_squared())) * n;
		return perp + parallel;
	}

	inline static float dot(const vec4& a, const vec4& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	inline static float max_comp(const vec4& v) {
		return std::max(v.x, std::max(v.y, std::max(v.z, v.w)));
	}

	inline static vec4 max(const vec4& v, float k) {
		return { std::max(v.x, k), std::max(v.y, k), std::max(v.z, k), std::max(v.w, k) };
	}

	inline static vec4 abs(const vec4& v) {
		return { std::abs(v.x), std::abs(v.y), std::abs(v.z), std::abs(v.w) };
	}

	inline static float length(const vec4& v) {
		return v.length();
	}
};

inline vec4 operator*(float k, const vec4& v) { return { k * v.x, k * v.y, k * v.z, k * v.w }; }

inline vec4 operator*(const vec4& v, float k) { return k * v; }

inline std::ostream& operator<<(std::ostream& out, const vec4& v) {
	out << v.x << " " << v.y << " " << v.z << " " << v.w;
	return out;
}


#endif