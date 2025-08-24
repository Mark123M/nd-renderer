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

// program
constexpr uint NUM_CPU_THREADS = 20;
constexpr float SIMULATION_RATE = 60.f;
constexpr float MIN_DELTA_TIME = 1.f / SIMULATION_RATE;

// rendering
constexpr float TOL = 5e-3f;
constexpr float EPSILON = 1e-3f;
constexpr float MAX_MARCH_DIST = 5.f;
constexpr uint MAX_MARCH_STEPS = 50;
constexpr uint SAMPLES_PER_PIXEL = 20;
constexpr uint MAX_RAY_BOUNCES = 20;

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

constexpr uint64_t multiplier = 6364136223846793005u;
constexpr uint64_t increment  = 1442695040888963407u;	// Or an arbitrary odd constant

__host__ __device__ uint32_t rotr32(uint32_t x, unsigned r) {
	return x >> r | x << (-r & 31);
}

__host__ __device__ uint32_t pcg32(uint64_t& state) {
	uint64_t x = state;
	unsigned count = (unsigned)(x >> 59);		// 59 = 64 - 5

	state = x * multiplier + increment;
	x ^= x >> 18;								// 18 = (64 - 27)/2
	return rotr32((uint32_t)(x >> 27), count);	// 27 = 32 - 5
}

__host__ __device__ void pcg32_init(uint64_t seed, uint64_t& state) {
	state = seed + increment;
	(void)pcg32(state);
}

__host__ __device__ float randf_pcg32(float min_val, float max_val, uint64_t& state) {
    // 1. Get a random 32-bit integer from PCG32
    uint32_t randint = pcg32(state);

    // 2. Normalize the integer to a float in [0.0, 1.0).
    // We multiply by the reciprocal of 2^32 for performance. Division is slower.
    // 2^32 is 4294967296.0f. Its reciprocal is approx 2.3283064e-10f.
    constexpr float norm = 1.f / 4294967296.f;
    float rand01 = randint * norm;

    // 3. Scale and shift to the desired range [min_val, max_val)
    return min_val + rand01 * (max_val - min_val);
}

#endif