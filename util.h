#ifndef UTIL_H
#define UTIL_H
#include <cuda_runtime.h>
#include <memory>

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

template <typename T>
struct managed_allocator {
   using value_type = T;

   managed_allocator() = default;
   template <class U>
   constexpr managed_allocator(const managed_allocator <U>&) noexcept {}

   T* allocate(std::size_t n) {
      T* ptr = nullptr;
      gpuErrchk(cudaMallocManaged(&ptr, n * sizeof(T)));

      if (!ptr) {
         throw std::bad_alloc();
      }

      return ptr;
   }

   void deallocate(T* p, std::size_t n) noexcept {
      gpuErrchk(cudaFree(p));
   }
};

template <typename T>
using managed_vector = std::vector<T, managed_allocator<T>>;

template<class T, class U>
bool operator==(const managed_allocator <T>&, const managed_allocator <U>&) { return true; }
 
template<class T, class U>
bool operator!=(const managed_allocator <T>&, const managed_allocator <U>&) { return false; }

struct managed_dtor {
   void operator()(void* ptr) const {
      if (ptr) {
         gpuErrchk(cudaFree(ptr));
      }
   }
};

template <typename T>
using managed_ptr = std::unique_ptr<T, managed_dtor>;

template <typename T>
managed_ptr<T> make_managed() {
   T* ptr = nullptr;
   gpuErrchk(cudaMallocManaged(&ptr, sizeof(T)));
   new (ptr) T;
   return managed_ptr<T>(ptr);
}

template <typename T, typename U>
__global__ void push_back_kernel(char* d_data, size_t size, T** d_list, size_t len, U object) {
   char* cur_object_data = d_data;
   for (size_t i = 0; i < len; i++) {
      T* cur_object_ptr = (T*)cur_object_data;
      d_list[i] = cur_object_ptr;
      cur_object_data += cur_object_ptr->size(); 
   }

   U* new_object_ptr = (U*)cur_object_data;
   new (new_object_ptr) U(object);
   d_list[len] = (T*)(new_object_ptr);
}

template <typename T>
__global__ void free_list_kernel(char* d_data, size_t len) {
   T* cur_object = (T*)d_data;
   for (size_t i = 0; i < len; i++) {
      cur_object->~T();
      cur_object += cur_object->size();
   }
}

// dynamic polymorphic list on device memory
// contiguous data buffer for better memory coalescing
template <typename T>
struct device_list {
   T** d_list;
   char* d_data;
   size_t len; // use doubling trick when scenes get much larger
   size_t data_size; 
   // use placement new for intrusive pointer, data, pointer, data layout? 
   device_list(): d_list{nullptr}, d_data{nullptr}, len{0}, data_size{0} {}

   template <typename U>
   void push_back(U object) {
      T** d_new_list;
      gpuErrchk(cudaMalloc(&d_new_list, (len + 1) * sizeof(T*)));
      gpuErrchk(cudaFree(d_list));
      d_list = d_new_list;

      char* d_new_data;
      gpuErrchk(cudaMalloc(&d_new_data, (data_size + sizeof(U)) * sizeof(char)));
      gpuErrchk(cudaMemcpy(d_new_data, d_data, data_size * sizeof(char), cudaMemcpyDeviceToDevice));
      gpuErrchk(cudaFree(d_data));
      d_data = d_new_data;

      push_back_kernel<T, U><<<1, 1>>>(d_data, data_size, d_list, len, object);
      gpuErrchk(cudaDeviceSynchronize());
      len++;
      data_size += sizeof(U);
   }

   __host__ __device__ T* operator[](int i) const {
      return d_list[i];
   }

   // used for stl container compatibilility
   size_t size() const {
      return len;
   }

   ~device_list() {
      free_list_kernel<T><<<1, 1>>>(d_data, len);
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaFree(d_list));
      gpuErrchk(cudaFree(d_data));
   }
};

// rendering
constexpr float TOL = 1e-4f;
constexpr float EPSILON = 1e-3f; // should be < TOL to avoid z-fighting
constexpr float MAX_RAY_DIST = 1000.f;
constexpr float MAX_MARCH_DIST = 5.f;
constexpr uint MAX_MARCH_STEPS = 50;
constexpr uint SAMPLES_PER_PIXEL = 20;
constexpr uint MAX_RAY_BOUNCES = 10;

// scene
constexpr float AMBIENT = 0.3f;
constexpr float AXIS_RADIUS = 0.01f;
constexpr float AXIS_LEN = 0.5f;
constexpr float CAMERA_MOVE_RATE = 2.f;
constexpr float CAMERA_ROTATE_RATE = 0.05f;
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