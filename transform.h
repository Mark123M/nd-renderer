#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "util.h"
#include "mat4.h"

struct transform {
	mat4 linear;
	mat4 inv_linear;
	vec4 translation;

	// standard basis
	transform(): linear{}, inv_linear{}, translation{} {}

	// construct from basis
	transform(const vec4& vx, const vec4& vy, const vec4& vz, const vec4& vw) {
		set_basis(vx, vy, vz, vw);
	}

	transform(const vec4& vx, const vec4& vy, const vec4& vz, const vec4& vw, const vec4& translation) {
		set_basis(vx, vy, vz, vw);
		set_translation(translation);
	}

	void set_basis(const vec4& vx, const vec4& vy, const vec4& vz, const vec4& vw) {
		assert(std::abs(vx.length_squared() - 1.f) <= TOL);
		assert(std::abs(vy.length_squared() - 1.f) <= TOL);
		assert(std::abs(vz.length_squared() - 1.f) <= TOL);
		assert(std::abs(vw.length_squared() - 1.f) <= TOL);

		linear = mat4{
			vx.x, vy.x, vz.x, vw.x,
			vx.y, vy.y, vz.y, vw.y,
			vx.z, vy.z, vz.z, vw.z,
			vx.w, vy.w, vz.w, vw.w
		};

		inv_linear = linear.transpose(); // orthogonal matrix
	}

	void set_translation(const vec4& translation) {
		this->translation = translation;
	}

	// simple rotation over the a-b plane
	void rotate(float angle, int a, int b) {
		assert(a < b);

		mat4 rot; // identity
		float cos_angle = cos(angle);
		float sin_angle = sin(angle);
		rot.m[a][a] = cos_angle;
		rot.m[b][b] = cos_angle;
		rot.m[a][b] = -sin_angle;
		rot.m[b][a] = sin_angle;

		linear = matmul(linear, rot);
		inv_linear = linear.transpose();
	}

	void rotate_xy(float angle) {
		rotate(angle, 0, 1);
	}

	void rotate_xz(float angle) {
		rotate(angle, 0, 2);
	}

	void rotate_xw(float angle) {
		rotate(angle, 0, 3);
	}

	void rotate_yz(float angle) {
		rotate(angle, 1, 2);
	}

	void rotate_yw(float angle) {
		rotate(angle, 1, 3);
	}

	void rotate_zw(float angle) {
		rotate(angle, 2, 3);
	}

	vec4 local_to_world(const vec4& v) const {
		return linear.vecmul(v) + translation; // Av + b
	}

	vec4 world_to_local(const vec4& w) const {
		return inv_linear.vecmul(w - translation); // A^-1(w - b)
	}
};

#endif // !TRANSFORM_H
