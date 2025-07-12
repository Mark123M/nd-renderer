#ifndef VEC4_H
#define VEC4_H

#include <cmath>
#include <iostream>
#include <cassert>

class vec4;
inline vec4 operator*(float k, const vec4& v);
inline float dot(const vec4& a, const vec4& b);

using point4 = vec4;

class vec4 {
public:
	float x, y, z, w;

	vec4() : x{ 0 }, y{ 0 }, z{ 0 }, w{ 0 } {}

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
		if (idx == 0) {
			return x;
		} else if (idx == 1) {
			return y;
		} else if (idx == 2) {
			return z;
		}

		return w;
	}

	void set(int idx, float val) {
		if (idx == 0) {
			x = val;
		} else if (idx == 1) {
			y = val;
		} else if (idx == 2) {
			z = val;
		}

		w = val;
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
};

inline vec4 operator*(float k, const vec4& v) { return { k * v.x, k * v.y, k * v.z, k * v.w }; }

inline vec4 operator*(const vec4& v, float k) { return k * v; }

inline std::ostream& operator<<(std::ostream& out, const vec4& v) {
	out << v.x << " " << v.y << " " << v.z << " " << v.w;
	return out;
}

inline vec4 normalize(const vec4& v) {
	return v / v.length();
}

inline vec4 reflect(const vec4& v, const vec4& n) {
	return v - 2 * dot(v, n) * n;
}

inline vec4 refract(const vec4& v, const vec4& n, float eta) {
	float cos_theta = std::fmin(dot(-v, n), 1.f);
	vec4 perp = eta * (v + cos_theta * n);
	vec4 parallel = -std::sqrt(std::fabs(1.f - perp.length_squared())) * n;
	return perp + parallel;
}

inline float dot(const vec4& a, const vec4& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

#endif