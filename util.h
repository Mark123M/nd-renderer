#ifndef UTIL_H
#define UTIL_H

constexpr float TOL = 1e-3f;
constexpr float EPSILON = 1e-3f;
constexpr float MAX_DIST = 100.f;
constexpr float AMBIENT = 0.3f;

const vec4 delta_x(EPSILON, 0.f, 0.f, 0.f);
const vec4 delta_y(0.f, EPSILON, 0.f, 0.f);
const vec4 delta_z(0.f, 0.f, EPSILON, 0.f);
const vec4 delta_w(0.f, 0.f, 0.f, EPSILON);

#endif