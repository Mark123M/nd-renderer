#ifndef MAT4_H
#define MAT4_H

#include "vec4.h"

struct mat4 {
	float m[4][4];

	mat4() {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				m[i][j] = (i == j) ? 1.0f : 0.0f;
			}
		}
	}

	/*
	float m00, m01, m02, m03;
	float m10, m11, m12, m13;
	float m20, m21, m22, m23;
	float m30, m31, m32, m33;
	*/
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	vec4 vecmul(const vec4& v) const {
		vec4 w;

		for (int i = 0; i < 4; i++) {
			float wi = 0;

			for (int k = 0; k < 4; k++) {
				wi += m[i][k] * v.get(k);
			}

			w.set(i, wi);
		}

		return w;
	}

	mat4 transpose() const {
		return {
			m[0][0], m[1][0], m[2][0], m[3][0],
			m[0][1], m[1][1], m[2][1], m[3][1],
			m[0][2], m[1][2], m[2][2], m[3][2],
			m[0][3], m[1][3], m[2][3], m[3][3]
		};
	}
};

inline mat4 matmul(const mat4& m1, const mat4& m2) {
	mat4 r;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float rij = 0;

			for (int k = 0; k < 4; k++) {
				rij += m1.m[i][k] * m2.m[k][j];
			}

			r.m[i][j] = rij;
		}
	}

	return r;
}

inline std::ostream& operator<<(std::ostream& out, const mat4& mat) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			out << mat.m[i][j] << " ";
		}
		out << std::endl;
	}

	return out;
}

#endif