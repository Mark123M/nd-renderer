#ifndef COLOR_H
#define COLOR_H

#include <fstream>
#include "vec4.h"

struct color {
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
};

inline float apply_gamma(float f) {
    if (f > 0) {
        return std::sqrt(f);
    }
    return 0;
}

inline void write_color(std::ofstream& file, const color& col) {
    // Red from left to right
    // Green from top to bottom
    int ir = (int)(std::min(1.f, apply_gamma(col.r)) * 255.999f);
    int ig = (int)(std::min(1.f, apply_gamma(col.g)) * 255.999f);
    int ib = (int)(std::min(1.f, apply_gamma(col.b)) * 255.999f);

    file << ir << " " << ig << " " << ib << " ";
}

#endif