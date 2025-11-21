#ifndef BVH_H
#define BVH_H

#include <algorithm>
#include "aabb.h"
#include <string>

struct bvh_node {
    aabb bbox;
    bvh_node* children[2];
    uint split_axis;
    uint L, R;

    void init_leaf(uint L0, uint R0, const aabb& b) {
        bbox = b;
        children[0] = children[1] = nullptr;
        L = L0;
        R = R0;
    }

    void init_interior(uint axis, bvh_node* c0, bvh_node* c1) {
        bbox = aabb::merge(c0->bbox, c1->bbox);
        children[0] = c0;
        children[1] = c1;
        split_axis = axis;
    }

    ~bvh_node() {
        delete children[0];
        delete children[1];
    }

	static std::string to_string(const bvh_node& node) {
        if (node.children[0] == nullptr && node.children[1] == nullptr) {
            return std::format("[LEAF] pmin {} | pmax {} | L {:d} | R {:d}", point4::to_string(node.bbox.p_min), point4::to_string(node.bbox.p_max), node.L, node.R);
        } else {
            return std::format("[INTERIOR] pmin {} | pmax {} | split_axis {:d}", point4::to_string(node.bbox.p_min), point4::to_string(node.bbox.p_max), node.split_axis);
        }
	}
};

struct bvh_bucket {
    uint count = 0;
    aabb bbox; 
};

struct linear_bvh_node {
    aabb bbox;
    uint second_child_idx;
    uint split_axis;
    uint L, R;
};

bvh_node* build_recursive(uint L, uint R, uint& num_nodes, std::vector<shape*>& shapes) {
    bvh_node* node = new bvh_node;
    num_nodes++;
    uint num_shapes = R - L + 1;

    aabb bbox;
    for (uint i = L; i <= R; i++) {
        bbox = aabb::merge(bbox, shapes[i]->bbox_world);
    }

    if (bbox.surface_volume() == 0.f || num_shapes == 1) {
        node->init_leaf(L, R, bbox);
        return node;
    }

    aabb bbox_centroid;
    for (uint i = L; i <= R; i++) {
        bbox_centroid = aabb::merge(bbox_centroid, shapes[i]->bbox_world.centroid());
    }

    uint dim = bbox_centroid.max_dim();
    if (bbox_centroid.p_max.get(dim) == bbox_centroid.p_min.get(dim)) {
        node->init_leaf(L, R, bbox);
        return node;
    }

    uint mid;
    if (num_shapes == 2) {
        mid = L;
        if (shapes[L]->bbox_world.centroid().get(dim) > shapes[R]->bbox_world.centroid().get(dim)) {
            std::swap(shapes[L], shapes[R]);
        }
    } else {
        std::vector<bvh_bucket> buckets(NUM_BVH_BUCKETS);

        for (uint i = L; i <= R; i++) {
            float offset = shapes[i]->bbox_world.centroid().get(dim) - bbox_centroid.p_min.get(dim);
            float extent = bbox_centroid.p_max.get(dim) - bbox_centroid.p_min.get(dim);
            uint b = NUM_BVH_BUCKETS * (offset / extent);
            b = min(b, NUM_BVH_BUCKETS - 1);

            buckets[b].count++;
            buckets[b].bbox = aabb::merge(buckets[b].bbox, shapes[i]->bbox_world);
        }

        std::vector<float> costs(NUM_BVH_BUCKETS - 1);
        uint count_below = 0;
        aabb bbox_below;
        for (uint i = 0; i < NUM_BVH_BUCKETS - 1; i++) {
            bbox_below = aabb::merge(bbox_below, buckets[i].bbox);
            count_below += buckets[i].count;
            costs[i] += count_below * bbox_below.surface_volume();
        }

        uint count_above = 0;
        aabb bbox_above;
        for (int i = NUM_BVH_BUCKETS - 1; i >= 1; i--) {
            bbox_above = aabb::merge(bbox_above, buckets[i].bbox);
            count_above += buckets[i].count;
            costs[i - 1] += count_above * bbox_above.surface_volume();
        }

        uint min_split_bucket = 0;
        float min_cost = __FLT_MAX__;
        for (int i = 0; i < NUM_BVH_BUCKETS - 1; i++) {
            if (costs[i] < min_cost) {
                min_cost = costs[i];
                min_split_bucket = i;
            }
        }

        float leaf_cost = num_shapes;
        min_cost = 1.f / 2.f + min_cost / bbox.surface_volume();

        if (min_cost < leaf_cost) {
            auto mid_iter = std::partition(
                shapes.begin() + L, shapes.begin() + R + 1,
                [=](const shape* bp) {
                    float offset = bp->bbox_world.centroid().get(dim) - bbox_centroid.p_min.get(dim);
                    float extent = bbox_centroid.p_max.get(dim) - bbox_centroid.p_min.get(dim);
                    int b = NUM_BVH_BUCKETS * (offset / extent);
                    b = min(b, NUM_BVH_BUCKETS - 1);
                    return b <= min_split_bucket;
                }
            );

            mid = (mid_iter - shapes.begin()) - 1;
            //UPDATE MID HERE?
        } else {
            node->init_leaf(L, R, bbox);
            return node;
        }
    }

    bvh_node* children[2];
    // ARE THESE INDICES CORRECT?
    children[0] = build_recursive(L, mid, num_nodes, shapes);
    children[1] = build_recursive(mid + 1, R, num_nodes, shapes);
    node->init_interior(dim, children[0], children[1]);
    return node;
}

void log_bvh(const bvh_node* root, uint depth) {
    if (root == nullptr) {
        return;
    }

    std::string spaces;
    for (int i = 0; i < depth; i++) {
        spaces += "     ";
    }
    std::string str = bvh_node::to_string(*root);
    std::cout << spaces << str << std::endl;
    
    log_bvh(root->children[0], depth + 1);
    log_bvh(root->children[1], depth + 1);
}

void build_bvh(std::vector<shape*>& shapes, std::vector<linear_bvh_node>& nodes) {
    uint num_nodes = 0;
    bvh_node* root = build_recursive(0, shapes.size() - 1, num_nodes, shapes);

    uint offset = 0;
    //flatten_bvh(root, &offset, nodes);
}

#endif