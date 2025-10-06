#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "util.h"
#include "affine.h"

struct transform {
	affine linear;
	affine inv_linear;

	__host__ __device__ void set_basis(const vec4& vx, const vec4& vy, const vec4& vz, const vec4& vw) {
		assert(approx_equals(vx.length_squared(), 1.f));
		assert(approx_equals(vy.length_squared(), 1.f));
		assert(approx_equals(vz.length_squared(), 1.f));
		assert(approx_equals(vw.length_squared(), 1.f));

		linear = affine{{
			{vx.x, vy.x, vz.x, vw.x, 0.f},
			{vx.y, vy.y, vz.y, vw.y, 0.f},
			{vx.z, vy.z, vz.z, vw.z, 0.f},
			{vx.w, vy.w, vz.w, vw.w, 0.f},
			{0.f,  0.f,  0.f,  0.f,  1.f}
		}};

		inv_linear = affine{{
			{vx.x, vx.y, vx.z, vx.w, 0.f},
			{vy.x, vy.y, vy.z, vy.w, 0.f},
			{vz.x, vz.y, vz.z, vz.w, 0.f},
			{vw.x, vw.y, vw.z, vw.w, 0.f},
			{0.f,  0.f,  0.f,  0.f,  1.f}
		}};
	}

	// simple rotation over the a-b plane
	__host__ __device__ void rotate(float angle, uint a, uint b) {
		affine R = rotate_mat(angle, a, b);
		linear = matmul(linear, R); // AR(R^-1A^-1)
		inv_linear = matmul(R.transpose(), inv_linear);
	}

	// R * T * R
	__host__ __device__ void translate(const vec4& t) {
		affine T = identity_affine;
		T.m[0][4] = t.x;
		T.m[1][4] = t.y;
		T.m[2][4] = t.z;
		T.m[3][4] = t.w;
		linear = matmul(T, linear); // TA(A^-1T^-1)

		affine T_inv = identity_affine;
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

	__host__ __device__ void rotate_around_point(float angle, const point4& p, uint a, uint b) {
		vec4 vecp = p - origin;
		translate(-vecp);
		rotate(angle, a, b);
		translate(vecp);
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

	__host__ __device__ point4 local_to_world(const point4& p) const {
		return linear.pointmul(p);
	}


	__host__ __device__ vec4 world_to_local(const vec4& w) const {
		return inv_linear.vecmul(w);
	}

	__host__ __device__ point4 world_to_local(const point4& p) const {
		return inv_linear.pointmul(p);
	}

	__host__ __device__ static affine rotate_mat(float angle, uint a, uint b) {
		assert(a < b);

		affine R = identity_affine; // identity
		float cos_angle = cosf(angle);
		float sin_angle = sinf(angle);
		R.m[a][a] = cos_angle;
		R.m[b][b] = cos_angle;
		R.m[a][b] = -sin_angle;
		R.m[b][a] = sin_angle;

		return R;
	}

	__host__ __device__ vec4 get_vec_x() const {
		return linear.get_x();
	}

	__host__ __device__ vec4 get_vec_y() const {
		return linear.get_y();
	}

	__host__ __device__ vec4 get_vec_z() const {
		return linear.get_z();
	}

	__host__ __device__ vec4 get_vec_w() const {
		return linear.get_w();
	}

	__host__ __device__ point4 get_pos() const {
		return linear.get_b();
	}

	__host__ __device__ void set_pos(const point4& pos) {
		return linear.set_b(pos);
	}

	__host__ __device__ static transform get_shading_transform(vec4& normal) {
		vec4 v[4];
		vec4 u[4];
		v[0] = normal;

		if (!approx_equals(normal.x, 0.f)) {
			v[1] = standard_y;
			v[2] = standard_z;
			v[3] = standard_w;
		} else if (!approx_equals(normal.y, 0.f)) {
			v[1] = standard_x;
			v[2] = standard_z;
			v[3] = standard_w;
		} else if (!approx_equals(normal.z, 0.f)) {
			v[1] = standard_x;
			v[2] = standard_y;
			v[3] = standard_w;
		} else {
			v[1] = standard_x;
			v[2] = standard_y;
			v[3] = standard_z;
		}

		// stable gram-schmidt algorithm
		for (int i = 0; i < 4; i++) {
			u[i] = v[i];
			u[i] = vec4::normalize(u[i]);
			
			for (int k = i + 1; k < 4; k++) {
				v[k] -= vec4::dot(v[k], u[i]) * u[i];
			}
		}

		transform t;
		t.set_basis(u[1], u[0], u[2], u[3]); // the normal should be the y-axis (up)
		return t;
	}
};

#endif // !TRANSFORM_H
