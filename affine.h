#ifndef AFFINE_H
#define AFFINE_H

#include "vec4.h"

struct affine {
	float m[5][5];

	/*
	float m00, m01, m02, m03;
	float m10, m11, m12, m13;
	float m20, m21, m22, m23;
	float m30, m31, m32, m33;
	*/
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
		return {{
			{m[0][0], m[1][0], m[2][0], m[3][0], 0.f},
			{m[0][1], m[1][1], m[2][1], m[3][1], 0.f},
			{m[0][2], m[1][2], m[2][2], m[3][2], 0.f},
			{m[0][3], m[1][3], m[2][3], m[3][3], 0.f},
			{0.f,     0.f,     0.f,     0.f,     1.f}
		}};
	}

	__host__ __device__ vec4 get_x() {
		return { m[0][0], m[1][0], m[2][0], m[3][0] };
	}

	__host__ __device__ vec4 get_y() {
		return { m[0][1], m[1][1], m[2][1], m[3][1] };
	}

	__host__ __device__ vec4 get_z() {
		return { m[0][2], m[1][2], m[2][2], m[3][2] };
	}

	__host__ __device__ vec4 get_w() {
		return { m[0][3], m[1][3], m[2][3], m[3][3] };
	}

	__host__ __device__ vec4 get_b() {
		return { m[0][4], m[1][4], m[2][4], m[3][4] };
    }

	__host__ __device__ void set_b(const vec4& b) {
        m[0][4] = b.x;
        m[1][4] = b.y;
        m[2][4] = b.z;
        m[3][4] = b.w;
	}

	__host__ __device__ void print() {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				printf("%.3f ", m[i][j]);
			}
			printf("\n");
		}

		printf("\n");
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

std::ostream& operator<<(std::ostream& out, const affine& mat) {
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			out << mat.m[i][j] << " ";
		}
		out << std::endl;
	}

	return out;
}

constexpr affine identity_affine {{
   {1.f, 0.f, 0.f, 0.f, 0.f},
   {0.f, 1.f, 0.f, 0.f, 0.f},
   {0.f, 0.f, 1.f, 0.f, 0.f},
   {0.f, 0.f, 0.f, 1.f, 0.f},
   {0.f, 0.f, 0.f, 0.f, 1.f}
}};

#endif