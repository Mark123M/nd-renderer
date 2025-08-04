#ifndef LIGHT_H
#define LIGHT_H

#include "color.h"
#include <cuda_runtime.h>

struct light {
	color col;

	__host__ __device__ light() : col{1.f, 1.f, 1.f} {}
	__host__ virtual color Le(point4 p, vec4 normal) = 0;
	__device__ virtual color Le_cuda(point4 p, vec4 normal) = 0;
	__device__ virtual void print_gpu() const = 0;
};

#endif // !DIRECTION_LIGHT_H
