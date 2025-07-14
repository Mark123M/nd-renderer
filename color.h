#ifndef COLOR_H
#define COLOR_H

#include <fstream>
#include "vec4.h"

struct color {
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;

    color(float r0, float g0, float b0): r{r0}, g{g0}, b{b0} {}

    color operator+(const color& other) {
        return { r + other.r, g + other.g, b + other.b };
    }

    color& operator+=(const color& other) {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }

    color operator*(const color& other) {
        return { r * other.r, g * other.g, b * other.b };
    }
};

inline color operator*(float k, color c) {
    return { k * c.r, k * c.g, k * c.b };
}

inline float apply_gamma(float f) {
    /* if (f > 0) {
        return std::sqrt(f);
    }
    return 0; */
    return f;
}

inline void write_color(std::ofstream& file, const color& col) {
    assert(col.r >= 0.f && col.g >= 0.f && col.b >= 0.f);

    int ir = (int)(std::min(1.f, apply_gamma(col.r)) * 255.999f);
    int ig = (int)(std::min(1.f, apply_gamma(col.g)) * 255.999f);
    int ib = (int)(std::min(1.f, apply_gamma(col.b)) * 255.999f);

    file << ir << " " << ig << " " << ib << " ";
}

#endif