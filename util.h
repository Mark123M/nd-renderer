#ifndef UTIL_H
#define UTIL_H
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// rendering
constexpr float TOL = 5e-3f;
constexpr float EPSILON = 1e-3f;
constexpr float MAX_MARCH_DIST = 5.f;
constexpr uint MAX_ITERS = 50;
constexpr uint NUM_CPU_THREADS = 20;
constexpr float SIMULATION_RATE = 60.f;
constexpr float MIN_DELTA_TIME = 1.f / SIMULATION_RATE;

// scene
constexpr float AMBIENT = 0.3f;
constexpr float AXIS_RADIUS = 0.01f;
constexpr float AXIS_LEN = 0.5f;
constexpr float CAMERA_MOVE_RATE = 2.f;
constexpr float CAMERA_ROTATE_RATE = 0.2f;
constexpr float ROTATE_RATE = 1.5f;

// math
constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385f;
constexpr float inv_pi = 0.31830988618379067154f;
constexpr float sqrt2 = 1.41421356237309504880f;
constexpr float pi_over_2 = 1.57079632679489661923f;
constexpr float pi_over_4 = 0.78539816339744830961f;

// utility functions
__host__ __device__ float deg2rad(float deg) {
	return deg * pi / 180.f;
}

#endif