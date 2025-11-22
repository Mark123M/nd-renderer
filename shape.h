#ifndef SHAPE_H
#define SHAPE_H

#include "vec4.h"
#include "color.h"
#include "transform.h"
#include "material.h"
#include "aabb.h"
#include <string>
#include <memory>

struct shape;

struct hit_result {
	point4 p;
	vec4 wo; // negative direction of ray
	vec4 normal; // always pointing outwards
	transform m;
	float t = MAX_RAY_DIST;
	const shape* target;
};

struct shape_sample {
	point4 p;
	float pdf = 1.f;
};

struct shape {
	transform basis;
	size_t mat_idx;

	__host__ __device__ shape() : basis{identity_affine, identity_affine} {}

	__host__ __device__ virtual float sdf(const point4& p) const {
		return MAX_RAY_DIST;
	}

	__host__ __device__ virtual bool intersect(const ray& r, hit_result& res) const {
		return false;
	}

	__host__ __device__ virtual float surface_volume() const {
		return 0.f; // not implemented
	}

	__host__ __device__ virtual bool sample(shape_sample& ss, const hit_result& res, uint64_t& pcg_state) const {
		return false;
	}

	__host__ __device__ virtual void rotate(float angle, uint a, uint b) {
		basis.rotate(angle, a, b);
	}
	
	__host__ __device__ virtual void translate(const vec4& t) {
		basis.translate(t);
	}

	__host__ __device__ void rotate_xy(float angle) {
		rotate(angle, 0, 1);
	}

	__host__ __device__ void rotate_xz(float angle) {
		rotate(angle, 0, 2);
	}

	__host__ __device__ void rotate_xw(float angle) {
		rotate(angle, 0, 3);
	}

	__host__ __device__ void rotate_yz(float angle) {
		rotate(angle, 1, 2);
	}

	__host__ __device__ void rotate_yw(float angle) {
		rotate(angle, 1, 3);
	}

	__host__ __device__ void rotate_zw(float angle) {
		rotate(angle, 2, 3);
	}

	__host__ __device__ virtual aabb get_bbox_local() const {
		return aabb();
	}

	__host__ __device__ virtual std::unique_ptr<shape> clone() const = 0;

	__host__ __device__ virtual size_t size() const = 0;

	__device__ virtual void print_gpu() const = 0;
};

struct shape_wrapper {
	std::string type;
	std::unique_ptr<shape> s;
	aabb bbox_world;

	shape_wrapper(const std::string& type, const shape& sh) : type{type}, s{sh.clone()}, bbox_world {} {
		update_bbox_world();
	}

	__host__ __device__ void update_bbox_world() {
		aabb bbox_local = s->get_bbox_local();
		point4& p_min_local = bbox_local.p_min;
		point4& p_max_local = bbox_local.p_max;
		point4 p_min_world = POINT4_MAX;
		point4 p_max_world = POINT4_MIN;
		
		for (uint x = 0; x < 2; x++) {
			for (uint y = 0; y < 2; y++) {
				for (uint z = 0; z < 2; z++) {
					for (uint w = 0; w < 2; w++) {
						point4 corner(
							x == 0 ? p_min_local.x : p_max_local.x,
							y == 0 ? p_min_local.y : p_max_local.y,
							z == 0 ? p_min_local.z : p_max_local.z,
							w == 0 ? p_min_local.w : p_max_local.w
						);
						point4 p = s->basis.local_to_world(corner);
						p_min_world = point4::min(p_min_world, p);
						p_max_world = point4::max(p_max_world, p);
					}
				}
			}
		}

		bbox_world.p_min = p_min_world;
		bbox_world.p_max = p_max_world;
	}
};

#endif