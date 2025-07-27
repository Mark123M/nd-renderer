#ifndef UTIL_H
#define UTIL_H

constexpr float TOL = 5e-3f;
constexpr float EPSILON = 1e-3f;
constexpr float MAX_DIST = 5.f;
constexpr float AMBIENT = 0.3f;

// Constants
constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385;
constexpr float inv_pi = 0.31830988618379067154;
constexpr float sqrt2 = 1.41421356237309504880;
constexpr float pi_over_2 = 1.57079632679489661923;
constexpr float pi_over_4 = 0.78539816339744830961;

// Utility functions
inline float deg2rad(float deg) {
	return deg * pi / 180.0;
}

const vec4 delta_x(EPSILON, 0.f, 0.f, 0.f);
const vec4 delta_y(0.f, EPSILON, 0.f, 0.f);
const vec4 delta_z(0.f, 0.f, EPSILON, 0.f);
const vec4 delta_w(0.f, 0.f, 0.f, EPSILON);

#endif