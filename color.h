#ifndef COLOR_H
#define COLOR_H

#include <fstream>
#include <cuda_runtime.h> // For make_float4, make_uchar4 etc.
#include "vec4.h"

struct color {
    float r;
    float g;
    float b;

    __host__ __device__ color(): r{0}, g{0}, b{0} {}

    __host__ __device__ color(float r0, float g0, float b0): r{r0}, g{g0}, b{b0} {}

    __host__ __device__ color operator+(const color& other) {
        return { r + other.r, g + other.g, b + other.b };
    }

    __host__ __device__ color& operator+=(const color& other) {
        r += other.r;
        g += other.g;
        b += other.b;

        return *this;
    }

    __host__ __device__ color operator*(const color& other) {
        return { r * other.r, g * other.g, b * other.b };
    }
};

__host__ __device__ color operator*(float k, color c) {
    return { k * c.r, k * c.g, k * c.b };
}

__host__ __device__ float apply_gamma(float f) {
    /* if (f > 0) {
        return std::sqrt(f);
    }
    return 0; */
    return f;
}

__host__ __device__ void write_color(std::ofstream& file, const color& col) {
    assert(col.r >= 0.f && col.g >= 0.f && col.b >= 0.f);

    int ir = (int)(fminf(1.f, apply_gamma(col.r)) * 255.999f);
    int ig = (int)(fminf(1.f, apply_gamma(col.g)) * 255.999f);
    int ib = (int)(fminf(1.f, apply_gamma(col.b)) * 255.999f);

    //file << ir << " " << ig << " " << ib << " ";
}

#endif