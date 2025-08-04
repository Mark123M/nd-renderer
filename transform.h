#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "math_util.h"
#include "affine.h"

struct transform {
	affine linear;
	affine inv_linear;

	// standard basis
	__host__ __device__ transform(): linear{}, inv_linear{} {}

	// construct from basis
	__host__ __device__ transform(const vec4& vx, const vec4& vy, const vec4& vz, const vec4& vw) {
		set_basis(vx, vy, vz, vw);
	}

	__host__ __device__ void set_basis(const vec4& vx, const vec4& vy, const vec4& vz, const vec4& vw) {
		assert(std::abs(vx.length_squared() - 1.f) <= TOL);
		assert(std::abs(vy.length_squared() - 1.f) <= TOL);
		assert(std::abs(vz.length_squared() - 1.f) <= TOL);
		assert(std::abs(vw.length_squared() - 1.f) <= TOL);

		linear = affine{
			vx.x, vy.x, vz.x, vw.x,
			vx.y, vy.y, vz.y, vw.y,
			vx.z, vy.z, vz.z, vw.z,
			vx.w, vy.w, vz.w, vw.w
		};

		inv_linear = affine(
			vx.x, vx.y, vx.z, vx.w,
			vy.x, vy.y, vy.z, vy.w,
			vz.x, vz.y, vz.z, vz.w,
			vw.x, vw.y, vw.z, vw.w
		);
	}

	// simple rotation over the a-b plane
	__host__ __device__ void rotate(float angle, int a, int b) {
		affine R = rotate_mat(angle, a, b);
		linear = matmul(linear, R); // RA(A^-1R^-1)
		inv_linear = matmul(R.transpose(), inv_linear);
	}

	// R * T * R
	__host__ __device__ void translate(const vec4& t) {
		affine T;
		T.m[0][4] = t.x;
		T.m[1][4] = t.y;
		T.m[2][4] = t.z;
		T.m[3][4] = t.w;
		linear = matmul(T, linear); // TA(A^-1T^-1)

		affine T_inv;
		T_inv.m[0][4] = -t.x;
		T_inv.m[1][4] = -t.y;
		T_inv.m[2][4] = -t.z;
		T_inv.m[3][4] = -t.w;
		inv_linear = matmul(inv_linear, T_inv);
	}

	__host__ __device__ void rotate_xy(float angle) {
		rotate(angle, 0, 1);
	}

	__host__ __device__ void rotate_xz(float angle) {
		rotate(angle, 0, 2);
	}

	__host__ __device__ void rotate_xw(float angle) {
		rotate(angle, 0, 3);
	}

	__host__ __device__ void rotate_yz(float angle) {
		rotate(angle, 1, 2);
	}

	__host__ __device__ void rotate_yw(float angle) {
		rotate(angle, 1, 3);
	}

	__host__ __device__ void rotate_zw(float angle) {
		rotate(angle, 2, 3);
	}

	__host__ __device__ void rotate_around_point(float angle, const point4& p, int a, int b) {
		translate(-p);
		rotate(angle, a, b);
		translate(p);
	}

	__host__ __device__ void rotate_xy_around_point(float angle, const point4& p) {
		rotate_around_point(angle, p, 0, 1);
	}

	__host__ __device__ void rotate_xz_around_point(float angle, const point4& p) {
		rotate_around_point(angle, p, 0, 2);
	}

	__host__ __device__ void rotate_xw_around_point(float angle, const point4& p) {
		rotate_around_point(angle, p, 0, 3);
	}

	__host__ __device__ void rotate_yz_around_point(float angle, const point4& p) {
		rotate_around_point(angle, p, 1, 2);
	}

	__host__ __device__ void rotate_yw_around_point(float angle, const point4& p) {
		rotate_around_point(angle, p, 1, 3);
	}

	__host__ __device__ void rotate_zw_around_point(float angle, const point4& p) {
		rotate_around_point(angle, p, 2, 3);
	}

	__host__ __device__ vec4 local_to_world(const vec4& v) const {
		return linear.vecmul(v);
	}

	__host__ __device__ vec4 world_to_local(const vec4& w) const {
		return inv_linear.vecmul(w);
	}

	__host__ __device__ static affine rotate_mat(float angle, int a, int b) {
		assert(a < b);

		affine R; // identity
		float cos_angle = cosf(angle);
		float sin_angle = sinf(angle);
		R.m[a][a] = cos_angle;
		R.m[b][b] = cos_angle;
		R.m[a][b] = -sin_angle;
		R.m[b][a] = sin_angle;

		return R;
	}
/*
	inline static vec4 rotate_vec(const vec4& v, float angle, int a, int b) {
		return rotate_mat(angle, a, b).vecmul(v);
	}

	inline static vec4 rotate_xy_vec(const vec4& v, float angle) {
		return rotate_vec(v, angle, 0, 1);
	}

	inline static vec4 rotate_xz_vec(const vec4& v, float angle) {
		return rotate_vec(v, angle, 0, 2);
	}

	inline static vec4 rotate_xw_vec(const vec4& v, float angle) {
		return rotate_vec(v, angle, 0, 3);
	}

	inline static vec4 rotate_yz_vec(const vec4& v, float angle) {
		return rotate_vec(v, angle, 1, 2);
	}

	inline static vec4 rotate_yw_vec(const vec4& v, float angle) {
		return rotate_vec(v, angle, 1, 3);
	}

	inline static vec4 rotate_zw_vec(const vec4& v, float angle) {
		return rotate_vec(v, angle, 2, 3);
	} */
};

#endif // !TRANSFORM_H
