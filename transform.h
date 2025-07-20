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
		mat4 R = rotate_mat(angle, a, b);
		linear = matmul(linear, R);
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

	void rotate_xy_around_point(float angle, const point4& p) {
		mat4 AR = matmul(linear, rotate_mat(angle, 0, 1));
		vec4 b_ = linear.vecmul(p) - AR.vecmul(p) + translation;
		linear = AR;
		inv_linear = linear.transpose();
		translation = b_;
		// set_translation(p - );
	}

	vec4 local_to_world(const vec4& v) const {
		return linear.vecmul(v) + translation; // Av + b
	}

	vec4 world_to_local(const vec4& w) const {
		return inv_linear.vecmul(w - translation); // A^-1(w - b)
	}

	inline static mat4 rotate_mat(float angle, int a, int b) {
		assert(a < b);

		mat4 R; // identity
		float cos_angle = cos(angle);
		float sin_angle = sin(angle);
		R.m[a][a] = cos_angle;
		R.m[b][b] = cos_angle;
		R.m[a][b] = -sin_angle;
		R.m[b][a] = sin_angle;

		return R;
	}

	inline static vec4 rotate_xy_vec(const vec4& v, float angle) {
		return rotate_mat(angle, 0, 1).vecmul(v);
	}

	inline static vec4 rotate_xz_vec(const vec4& v, float angle) {
		return rotate_mat(angle, 0, 2).vecmul(v);
	}

	inline static vec4 rotate_xw_vec(const vec4& v, float angle) {
		return rotate_mat(angle, 0, 3).vecmul(v);
	}

	inline static vec4 rotate_yz_vec(const vec4& v, float angle) {
		return rotate_mat(angle, 1, 2).vecmul(v);
	}

	inline static vec4 rotate_yw_vec(const vec4& v, float angle) {
		return rotate_mat(angle, 1, 3).vecmul(v);
	}

	inline static vec4 rotate_zw_vec(const vec4& v, float angle) {
		return rotate_mat(angle, 2, 3).vecmul(v);
	}
};

#endif // !TRANSFORM_H
