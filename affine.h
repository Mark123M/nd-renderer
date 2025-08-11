#ifndef AFFINE_H
#define AFFINE_H

#include "vec4.h"

struct affine {
	float m[5][5];

	__host__ __device__ affine() {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				m[i][j] = (i == j) ? 1.f : 0.f;
			}
		}
	}

	/*
	float m00, m01, m02, m03;
	float m10, m11, m12, m13;
	float m20, m21, m22, m23;
	float m30, m31, m32, m33;
	*/
	__host__ __device__ affine(float m00, float m01, float m02, float m03,
		 float m10, float m11, float m12, float m13,
		 float m20, float m21, float m22, float m23,
		 float m30, float m31, float m32, float m33
        ) {                                                         // b: translation vector
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03; m[0][4] = 0.f;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13; m[1][4] = 0.f;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23; m[2][4] = 0.f;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33; m[3][4] = 0.f;
        m[4][0] = 0.f;   m[4][1] = 0.f;   m[4][2] = 0.f;   m[4][3] = 0.f;   m[4][4] = 1.f;
	}

	__host__ __device__ vec4 vecmul(const vec4& v) const {
		vec4 w;

		for (int i = 0; i < 4; i++) {
			float wi = 0.f;

			for (int k = 0; k < 4; k++) {
				wi += m[i][k] * v.get(k);
			}

			w.set(i, wi + m[i][4]);
		}

		return w;
	}

	__host__ __device__ affine transpose() const {
		return {
			m[0][0], m[1][0], m[2][0], m[3][0],
			m[0][1], m[1][1], m[2][1], m[3][1],
			m[0][2], m[1][2], m[2][2], m[3][2],
			m[0][3], m[1][3], m[2][3], m[3][3]
		};
	}

    __host__ __device__ void set_b(const vec4& b) {
        m[0][4] = b.x;
        m[1][4] = b.y;
        m[2][4] = b.z;
        m[3][4] = b.w;
    }
};

__host__ __device__ affine matmul(const affine& m1, const affine& m2) {
	affine r;

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			float rij = 0.f;

			for (int k = 0; k < 5; k++) {
				rij += m1.m[i][k] * m2.m[k][j];
			}

			r.m[i][j] = rij;
		}
	}

	return r;
}

__host__ std::ostream& operator<<(std::ostream& out, const affine& mat) {
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			out << mat.m[i][j] << " ";
		}
		out << std::endl;
	}

	return out;
}

#endif