#ifndef NCUBE_H
#define NCUBE_H

#include "vec4.h"
#include "color.h"
#include "shape.h"

// Multiplies a 4x4 matrix with a vec4 'p'
inline vec4 multiply_matrix_vec4(const float m[4][4], const vec4& p) {
    vec4 result;
    result.x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3] * p.w;
    result.y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3] * p.w;
    result.z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3] * p.w;
    result.w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3] * p.w;
    return result;
}

struct ncube : public shape {
	// CENTER IS ALWAYS (0, 0, 0, 0) FOR NOW
	point4 corner;

	ncube(const point4& corner, int n = 4) : shape{n}, corner{ corner } {}
	ncube(const point4& corner, const color& albedo0, int n = 4) : shape{ albedo0, n }, corner{ corner } {}

	float sdf(const point4& p) const override {

        // Example usage with the given matrix
        float example_matrix[4][4] = {
            { -0.0971147f,  -0.30548878f, -0.63968163f, -0.69860772f },
            {  0.80357565f, -0.51840114f,  0.26369012f, -0.12646725f },
            {  0.58114334f,  0.58723162f, -0.5408912f,   0.15769641f },
            { -0.08430502f, -0.54138331f, -0.47823806f,  0.68635641f }
        };

        point4 pp = multiply_matrix_vec4(example_matrix, p);
		
		vec4 q = abs(pp) - corner;
		float d = max(q, 0).length() + std::min(max_comp(q), 0.f);
		return d;
	}
};

#endif