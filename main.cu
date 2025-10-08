#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <thread>

#include "color.h"
#include "ray.h"
#include "nsphere.h"
#include "sphere.h"
#include "ncube.h"
#include "cylinder.h"
#include "quad.h"
#include "cube.h"
#include "projected_cylinder.h"
#include "camera.h"
#include "world.h"
#include "direction_light.h"
#include "lambertian.h"
#include "dielectric.h"
#include "light_material.h"
#include "util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <nvtx3/nvToolsExt.h>

uint g_sm_count;
uint g_sm_max_threads;
float fixed_delta_time = 0.f;

struct cudaGraphicsResource* render_texture_CUDA = nullptr;

__global__ void test_render_kernel(uchar4* d_image_data, uint width, uint height, float time)
{
    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        // Normalize coordinates to [0, 1]
        float nx = (float)row / height;
        float ny = (float)col / width;

        // Simple animation logic
        float r = 0.5f + 0.5f * sinf(nx * 10.0f + time);
        float g = 0.5f + 0.5f * sinf(ny * 10.0f + time + 2.0f);
        float b = 0.5f + 0.5f * sinf((nx + ny) * 5.0f + time + 4.0f);

        // Map to 0-255 range
        uchar4 pixel;
        pixel.x = static_cast<unsigned char>(r * 255.0f); // Red
        pixel.y = static_cast<unsigned char>(g * 255.0f); // Green
        pixel.z = static_cast<unsigned char>(b * 255.0f); // Blue
        pixel.w = 255;                                    // Alpha (fully opaque)

        // Write to the linear device memory buffer
        d_image_data[row * width + col] = pixel;
    }
}

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

__global__ void hello() {
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

static bool handle_inputs_cuda() {
    bool did_input = false;
    uint threads_per_block = 64;
    uint num_blocks = (scene.size() + threads_per_block - 1) / threads_per_block;

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 mouse_delta = io.MouseDelta;
    
    if (ImGui::IsKeyDown(ImGuiKey_Space)) {
        if (mouse_delta.x != 0.0f) {
            world::rotate_camera_horizontal_kernel<<<1, 1>>>(mouse_delta.x * CAMERA_ROTATE_RATE * fixed_delta_time);
        }
        if (mouse_delta.y != 0.0f) {
            world::rotate_camera_vertical_kernel<<<1, 1>>>(mouse_delta.y * CAMERA_ROTATE_RATE * fixed_delta_time);
        }

        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_Z)) {
        if (mouse_delta.x != 0.f) {
            world::rotate_camera_yw_kernel<<<1, 1>>>(mouse_delta.x * CAMERA_ROTATE_RATE * fixed_delta_time);
        }

        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_C)) {
        if (mouse_delta.x != 0.f) {
            world::rotate_camera_xw_kernel<<<1, 1>>>(mouse_delta.x * CAMERA_ROTATE_RATE * fixed_delta_time);
        }

        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_A)) {
        //world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(-MOVE_RATE * fixed_delta_time, 0.f, 0.f, 0.f));
        world::move_camera_x_kernel<<<1, 1>>>(-CAMERA_MOVE_RATE * fixed_delta_time);
        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_D)) {
        //world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(MOVE_RATE * fixed_delta_time, 0.f, 0.f, 0.f));
        world::move_camera_x_kernel<<<1, 1>>>(CAMERA_MOVE_RATE * fixed_delta_time);
        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_W)) {
        //world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(0.f, MOVE_RATE * fixed_delta_time, 0.f, 0.f));
        world::move_camera_z_kernel<<<1, 1>>>(-CAMERA_MOVE_RATE * fixed_delta_time);
        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_S)) {
        //world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(0.f, -MOVE_RATE * fixed_delta_time, 0.f, 0.f));
        world::move_camera_z_kernel<<<1, 1>>>(CAMERA_MOVE_RATE * fixed_delta_time);
        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_Q)) {
        world::move_camera_w_kernel<<<1, 1>>>(CAMERA_MOVE_RATE * fixed_delta_time);
        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_E)) {
        world::move_camera_w_kernel<<<1, 1>>>(-CAMERA_MOVE_RATE * fixed_delta_time);
        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) {
        world::rotate_xz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_RATE * fixed_delta_time, scene.d_list);
        world::rotate_yw_kernel<<<num_blocks, threads_per_block>>>(ROTATE_RATE * fixed_delta_time, scene.d_list);
        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) {
        world::rotate_yz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_RATE * fixed_delta_time, scene.d_list);
        did_input = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) {
        world::rotate_xz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_RATE * fixed_delta_time, scene.d_list);
        did_input = true;
    }

    return did_input;
}

static float handle_inputs() {
    bool did_input = false;

    if (ImGui::IsKeyPressed(ImGuiKey_A)) {
        for (size_t i = 0; i < scene.size(); i++) {
            shape* c = scene[i];
            c->translate(vec4(-CAMERA_MOVE_RATE * fixed_delta_time, 0.f, 0.f, 0.f));
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_D)) {
        for (size_t i = 0; i < scene.size(); i++) {
            shape* c = scene[i];
            c->translate(vec4(CAMERA_MOVE_RATE * fixed_delta_time, 0.f, 0.f, 0.f));
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_W)) {
        for (size_t i = 0; i < scene.size(); i++) {
            shape* c = scene[i];
            c->translate(vec4(0.f, CAMERA_MOVE_RATE * fixed_delta_time, 0.f, 0.f));
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_S)) {
        for (size_t i = 0; i < scene.size(); i++) {
            shape* c = scene[i];
            c->translate(vec4(0.f, -CAMERA_MOVE_RATE * fixed_delta_time, 0.f, 0.f));
        }
        
        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        for (size_t i = 0; i < scene.size(); i++) {
            shape* c = scene[i];
            c->rotate_xz(ROTATE_RATE * fixed_delta_time);
            c->rotate_yw(ROTATE_RATE * fixed_delta_time);
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
        for (size_t i = 0; i < scene.size(); i++) {
            shape* c = scene[i];
            c->rotate_yz(ROTATE_RATE * fixed_delta_time);
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        for (size_t i = 0; i < scene.size(); i++) {
            shape* c = scene[i];
            c->rotate_xz(ROTATE_RATE * fixed_delta_time);
        }

        did_input = true;
    }

    return did_input;
}


__global__ void math_test() {
    vec4 N(-1.4f, 0.5f, -3.f, 2.f);
    transform t = transform::get_shading_transform(N);
    vec4 vx = t.get_vec_x(), vy = t.get_vec_y(), vz = t.get_vec_z(), vw = t.get_vec_w();
    printf("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", vec4::dot(vx, vy), vec4::dot(vx, vz), vec4::dot(vx, vw), vec4::dot(vy, vz), vec4::dot(vy, vw), vec4::dot(vz, vw), vec4::length(vx), vec4::length(vy), vec4::length(vz), vec4::length(vw));
    t.linear.print();

    vec4 N2(1.f, 0.f, 0.f, 0.f);
    transform t2 = transform::get_shading_transform(N2);
    t2.linear.print();
}

void tesseract_lines() {
        // Tesseract Scene
    float L = 0.5f;

    point4 vertices[16] = {
        // Vertex Index corresponds to binary (WZYX) -> e.g., 0000 for (0,0,0,0), 1111 for (L,L,L,L)

        // --- Vertices where W = 0.0f ---
        // Z = 0.0f
        point4(0.0f, 0.0f, 0.0f, 0.0f), // 0: (0,0,0,0)
        point4(L,    0.0f, 0.0f, 0.0f), // 1: (L,0,0,0)
        point4(0.0f, L,    0.0f, 0.0f), // 2: (0,L,0,0)
        point4(L,    L,    0.0f, 0.0f), // 3: (L,L,0,0)
        // Z = L
        point4(0.0f, 0.0f, L,    0.0f), // 4: (0,0,L,0)
        point4(L,    0.0f, L,    0.0f), // 5: (L,0,L,0)
        point4(0.0f, L,    L,    0.0f), // 6: (0,L,L,0)
        point4(L,    L,    L,    0.0f), // 7: (L,L,L,0)

        // --- Vertices where W = L (0.5f) ---
        // Z = 0.0f
        point4(0.0f, 0.0f, 0.0f, L),    // 8: (0,0,0,L)
        point4(L,    0.0f, 0.0f, L),    // 9: (L,0,0,L)
        point4(0.0f, L,    0.0f, L),    // 10: (0,L,0,L)
        point4(L,    L,    0.0f, L),    // 11: (L,L,0,L)
        // Z = L
        point4(0.0f, 0.0f, L,    L),    // 12: (0,0,L,L)
        point4(L,    0.0f, L,    L),    // 13: (L,0,L,L)
        point4(0.0f, L,    L,    L),    // 14: (0,L,L,L)
        point4(L,    L,    L,    L)     // 15: (L,L,L,L)
    };
    
    //point4 center(L / 2, L / 2, L / 2, L / 2);

    uint edges[32][2] = {
        // --- Edges within the W=0.0f 'cube' (indices 0-7) ---
        {0, 1}, // (0,0,0,0) to (L,0,0,0) - X-axis
        {0, 2}, // (0,0,0,0) to (0,L,0,0) - Y-axis
        {0, 4}, // (0,0,0,0) to (0,0,L,0) - Z-axis

        {1, 3},
        {1, 5},

        {2, 3},
        {2, 6},

        {3, 7},

        {4, 5},
        {4, 6},

        {5, 7},

        {6, 7},

        // --- Edges within the W=L 'cube' (indices 8-15) ---
        {8, 9}, // (0,0,0,L) to (L,0,0,L) - X-axis
        {8, 10}, // (0,0,0,L) to (0,L,0,L) - Y-axis
        {8, 12}, // (0,0,0,L) to (0,0,L,L) - Z-axis

        {9, 11},
        {9, 13},

        {10, 11},
        {10, 14},

        {11, 15},

        {12, 13},
        {12, 14},

        {13, 15},

        {14, 15},

        // --- Edges connecting the W=0.0f 'cube' to the W=L 'cube' (W-axis edges) ---
        {0, 8},  // (0,0,0,0) to (0,0,0,L)
        {1, 9},  // (L,0,0,0) to (L,0,0,L)
        {2, 10}, // (0,L,0,0) to (0,L,0,L)
        {3, 11}, // (L,L,0,0) to (L,L,0,L)
        {4, 12}, // (0,0,L,0) to (0,0,L,L)
        {5, 13}, // (L,0,L,0) to (L,0,L,L)
        {6, 14}, // (0,L,L,0) to (0,L,L,L)
        {7, 15}  // (L,L,L,0) to (L,L,L,L)
    };

    color edge_color(1.0f, 0.647f, 0.0f); // Orange
    lambertian edge_mat(edge_color);
    materials.push_back(edge_mat);

    for (int i = 0; i < 32; ++i) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        vec4 move(L / 2, L / 2, L / 2, L / 2);
        projected_cylinder pc;
        pc.start0 = vertices[idx1] - move;
        pc.end0 = vertices[idx2] - move;
        pc.mat_idx = 0;
        scene.push_back(pc);
    }

    /*std::unique_ptr<ncube> nc = std::make_unique<ncube>();
    point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    nc->corner = cor;
    nc->albedo = edge_color;
    scene.push_back(nc.get()); */

    /*std::unique_ptr<nsphere> ball = std::make_unique<nsphere>();
    ball->radius = 0.5f;
    ball->albedo = color(1.f, 0.f, 0.f);
    scene.push_back(ball.get());*/
    direction_light lig1;
    lig1.col = color(1.0f, 1.0f, 0.9f);
    lig1.dir = vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f));
    lights.push_back(lig1);
}

void tesseract_lines_reflector() {
        // Tesseract Scene
    float L = 0.5f;

    point4 vertices[16] = {
        // Vertex Index corresponds to binary (WZYX) -> e.g., 0000 for (0,0,0,0), 1111 for (L,L,L,L)

        // --- Vertices where W = 0.0f ---
        // Z = 0.0f
        point4(0.0f, 0.0f, 0.0f, 0.0f), // 0: (0,0,0,0)
        point4(L,    0.0f, 0.0f, 0.0f), // 1: (L,0,0,0)
        point4(0.0f, L,    0.0f, 0.0f), // 2: (0,L,0,0)
        point4(L,    L,    0.0f, 0.0f), // 3: (L,L,0,0)
        // Z = L
        point4(0.0f, 0.0f, L,    0.0f), // 4: (0,0,L,0)
        point4(L,    0.0f, L,    0.0f), // 5: (L,0,L,0)
        point4(0.0f, L,    L,    0.0f), // 6: (0,L,L,0)
        point4(L,    L,    L,    0.0f), // 7: (L,L,L,0)

        // --- Vertices where W = L (0.5f) ---
        // Z = 0.0f
        point4(0.0f, 0.0f, 0.0f, L),    // 8: (0,0,0,L)
        point4(L,    0.0f, 0.0f, L),    // 9: (L,0,0,L)
        point4(0.0f, L,    0.0f, L),    // 10: (0,L,0,L)
        point4(L,    L,    0.0f, L),    // 11: (L,L,0,L)
        // Z = L
        point4(0.0f, 0.0f, L,    L),    // 12: (0,0,L,L)
        point4(L,    0.0f, L,    L),    // 13: (L,0,L,L)
        point4(0.0f, L,    L,    L),    // 14: (0,L,L,L)
        point4(L,    L,    L,    L)     // 15: (L,L,L,L)
    };
    
    //point4 center(L / 2, L / 2, L / 2, L / 2);

    uint edges[32][2] = {
        // --- Edges within the W=0.0f 'cube' (indices 0-7) ---
        {0, 1}, // (0,0,0,0) to (L,0,0,0) - X-axis
        {0, 2}, // (0,0,0,0) to (0,L,0,0) - Y-axis
        {0, 4}, // (0,0,0,0) to (0,0,L,0) - Z-axis

        {1, 3},
        {1, 5},

        {2, 3},
        {2, 6},

        {3, 7},

        {4, 5},
        {4, 6},

        {5, 7},

        {6, 7},

        // --- Edges within the W=L 'cube' (indices 8-15) ---
        {8, 9}, // (0,0,0,L) to (L,0,0,L) - X-axis
        {8, 10}, // (0,0,0,L) to (0,L,0,L) - Y-axis
        {8, 12}, // (0,0,0,L) to (0,0,L,L) - Z-axis

        {9, 11},
        {9, 13},

        {10, 11},
        {10, 14},

        {11, 15},

        {12, 13},
        {12, 14},

        {13, 15},

        {14, 15},

        // --- Edges connecting the W=0.0f 'cube' to the W=L 'cube' (W-axis edges) ---
        {0, 8},  // (0,0,0,0) to (0,0,0,L)
        {1, 9},  // (L,0,0,0) to (L,0,0,L)
        {2, 10}, // (0,L,0,0) to (0,L,0,L)
        {3, 11}, // (L,L,0,0) to (L,L,0,L)
        {4, 12}, // (0,0,L,0) to (0,0,L,L)
        {5, 13}, // (L,0,L,0) to (L,0,L,L)
        {6, 14}, // (0,L,L,0) to (0,L,L,L)
        {7, 15}  // (L,L,L,0) to (L,L,L,L)
    };

    color edge_color(1.0f, 0.647f, 0.0f); // Orange
    lambertian edge_mat(edge_color);
    materials.push_back(edge_mat);

    for (int i = 0; i < 32; ++i) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        vec4 move(L / 2, L / 2, L / 2, L / 2);
        projected_cylinder pc;
        pc.start0 = vertices[idx1] - move;
        pc.end0 = vertices[idx2] - move;
        pc.mat_idx = 0;
        scene.push_back(pc);
    }

    color sphere_color(0.f, 1.f, 0.f);
    specular sphere_mat(sphere_color);
    materials.push_back(sphere_mat);

    nsphere ns;
    ns.radius = 0.25f;
    ns.mat_idx = 1;
    //ns->translate(vec4(-1.f, 0.f, -1.f, 0.f));
    scene.push_back(ns);

    /*std::unique_ptr<ncube> nc = std::make_unique<ncube>();
    point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    nc->corner = cor;
    nc->albedo = edge_color;
    scene.push_back(nc.get()); */

    /*std::unique_ptr<nsphere> ball = std::make_unique<nsphere>();
    ball->radius = 0.5f;
    ball->albedo = color(1.f, 0.f, 0.f);
    scene.push_back(ball.get());*/
    direction_light lig1;
    lig1.col = color(1.0f, 1.0f, 0.9f);
    lig1.dir = vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f));
    lights.push_back(lig1);
}

void tesseract() {
    color edge_color(1.0f, 0.647f, 0.0f); // Orange
    lambertian edge_mat(edge_color);
    materials.push_back(edge_mat);

    color sphere_color(0.f, 1.f, 0.f);
    specular sphere_mat(sphere_color);
    materials.push_back(sphere_mat);

    color floor_color(0.5f, 0.5f, 0.5f);
    lambertian floor_mat(floor_color);
    materials.push_back(floor_mat);

    ncube nc;
    // point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    // nc->corner = cor;
    nc.mat_idx = 0;
    scene.push_back(nc);

    nsphere ns;
    ns.radius = 0.5f;
    ns.mat_idx = 1;
    ns.translate(vec4(-1.f, 0.f, -1.f, 0.f));
    scene.push_back(ns);

    nsphere floor;
    floor.radius = 100.f;
    floor.mat_idx = 2;
    floor.translate(vec4(0.f, -100.5f, -1.f, 0.f));
    scene.push_back(floor);

    direction_light lig1;
    lig1.col = color(1.0f, 1.0f, 0.9f);
    lig1.dir = vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f));
    lights.push_back(lig1);
}

void all_white() {
    color edge_color(1.f, 1.f, 1.f); // Orange
    lambertian edge_mat(edge_color);
    materials.push_back(edge_mat);

    color sphere_color(1.f, 1.f, 1.f);
    specular sphere_mat(sphere_color);
    materials.push_back(sphere_mat);

    color floor_color(1.f, 1.f, 1.f);
    lambertian floor_mat(floor_color);
    materials.push_back(floor_mat);

    ncube nc;
    // point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    // nc->corner = cor;
    nc.mat_idx = 0;
    scene.push_back(nc);

    nsphere ns;
    ns.radius = 0.5f;
    ns.mat_idx = 1;
    ns.translate(vec4(-1.f, 0.f, -1.f, 0.f));
    scene.push_back(ns);

    nsphere floor;
    floor.radius = 100.f;
    floor.mat_idx = 2;
    floor.translate(vec4(0.f, -100.5f, -1.f, 0.f));
    scene.push_back(floor);

    direction_light lig1;
    lig1.col = color(1.0f, 1.0f, 0.9f);
    lig1.dir = vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f));
    lights.push_back(lig1);
}

void tesseract_reflector() {
    color edge_color(1.0f, 0.647f, 0.0f); // Orange
    specular edge_mat(edge_color);
    materials.push_back(edge_mat);

    color sphere_color(0.f, 1.f, 0.f);
    specular sphere_mat(sphere_color);
    materials.push_back(sphere_mat);

    color floor_color(0.5f, 0.5f, 0.5f);
    lambertian floor_mat(floor_color);
    materials.push_back(floor_mat);

    ncube nc;
    // point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    // nc->corner = cor;
    nc.mat_idx = 0;
    scene.push_back(nc);

    nsphere ns;
    ns.radius = 0.5f;
    ns.mat_idx = 1;
    ns.translate(vec4(-1.f, 0.f, -1.f, 0.f));
    scene.push_back(ns);

    nsphere floor;
    floor.radius = 100.f;
    floor.mat_idx = 2;
    floor.translate(vec4(0.f, -100.5f, -1.f, 0.f));
    scene.push_back(floor);

    direction_light lig1;
    lig1.col = color(1.0f, 1.0f, 0.9f);
    lig1.dir = vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f));
    lights.push_back(lig1);
}

void tesseract_glass() {
    smooth_dielectric edge_mat(1.5f);
    materials.push_back(edge_mat);

    color sphere_color(0.f, 1.f, 0.f);
    specular sphere_mat(sphere_color);
    materials.push_back(sphere_mat);

    color floor_color(1.f, 0.8f, 0.5f);
    lambertian floor_mat(floor_color);
    materials.push_back(floor_mat);

    ncube nc;
    // point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    // nc->corner = cor;
    nc.mat_idx = 0;
    scene.push_back(nc);

    nsphere ns;
    ns.radius = 0.5f;
    ns.mat_idx = 1;
    ns.translate(vec4(-1.f, 0.f, -1.f, 0.f));
    scene.push_back(ns);

    nsphere floor;
    floor.radius = 100.f;
    floor.mat_idx = 2;
    floor.translate(vec4(0.f, -100.5f, -1.f, 0.f));
    scene.push_back(floor);

    direction_light lig1;
    lig1.col = color(1.0f, 1.0f, 0.9f);
    lig1.dir = vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f));
    lights.push_back(lig1);
}

void spheres() {
    color sphere_color(0.9f, 0.5f, 0.9f);
    specular sphere_mat(sphere_color);
    materials.push_back(sphere_mat);

    color sphere2_color(0.f, 1.f, 1.f);
    specular sphere2_mat(sphere2_color);
    materials.push_back(sphere2_mat);

    smooth_dielectric sphere3_mat(1.5f);
    materials.push_back(sphere3_mat);

    color floor_color(0.5f, 0.5f, 0.5f);
    lambertian floor_mat(floor_color);
    materials.push_back(floor_mat);

    nsphere ns;
    ns.radius = 0.5f;
    ns.mat_idx = 0;
    ns.translate(vec4(-1.f, 0.f, -1.f, 0.f));
    scene.push_back(ns);

    nsphere ns2;
    ns2.radius = 0.5f;
    ns2.mat_idx = 1;
    ns2.translate(vec4(0.f, 0.f, -1.f, 0.f));
    scene.push_back(ns2);

    nsphere ns3;
    ns3.radius = 0.5f;
    ns3.mat_idx = 2;
    ns3.translate(vec4(1.f, 0.f, -1.f, 0.f));
    scene.push_back(ns3);

    nsphere floor;
    floor.radius = 100.f;
    floor.mat_idx = 3;
    floor.translate(vec4(0.f, -100.5f, -1.f, 0.f));
    scene.push_back(floor);

    direction_light lig1;
    lig1.col = color(1.0f, 1.0f, 0.9f);
    lig1.dir = vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f));
    lights.push_back(lig1);
}

// BSDFs are modelled with the unit half 3-sphere (if there is a 4-th spatial dimension it makes sense?)
void quad_light_test() {
    lambertian red(color(1.f, 1.f, 0.05f));
    materials.push_back(red);
    light_material white(color(4.f, 4.f, 4.f));
    materials.push_back(white);
    lambertian green(color(0.12f, 0.45f, 0.15f));
    materials.push_back(green);

    quad q1(point4(-3.f, -2.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q1.mat_idx = 0;
    scene.push_back(q1);

    quad q2(point4(-2.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -1.f, 0.f), vec4(0.f, 1.f, 0.f, 0.f));
    q2.mat_idx = 1;
    scene.push_back(q2);
}

void cornell_box() {
    lambertian red(color(0.65f, 0.05f, 0.05f));
    lambertian white(color(0.73f, 0.73f, 0.73f));
    lambertian green(color(0.12f, 0.45f, 0.15f));
    light_material ceiling(color(15.f, 15.f, 15.f));
    materials.push_back(red);
    materials.push_back(white);
    materials.push_back(green);
    materials.push_back(ceiling);

    quad q1(point4(555.f, 0.f, 0.f, 0.f), vec4(0.f, 555.f, 0.f, 0.f), vec4(0.f, 0.f, 555.f, 0.f));
    q1.mat_idx = 2;
    quad q2(point4(0.f, 0.f, 0.f, 0.f), vec4(0.f, 555.f, 0.f, 0.f), vec4(0.f, 0.f, 555.f, 0.f));
    q2.mat_idx = 0;
    quad q3(point4(343.f, 554.f, 332.f, 0.f), vec4(-130.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -105.f, 0.f));
    q3.mat_idx = 3;
    quad q4(point4(0.f, 0.f, 0.f, 0.f), vec4(555.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, 555.f, 0.f));
    q4.mat_idx = 1;
    quad q5(point4(555.f, 555.f, 555.f, 0.f), vec4(-555.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -555.f, 0.f));
    q5.mat_idx = 1;
    quad q6(point4(0.f, 0.f, 555.f, 0.f), vec4(555.f, 0.f, 0.f, 0.f), vec4(0.f, 555.f, 0.f, 0.f));
    q6.mat_idx = 1;
    scene.push_back(q1);
    scene.push_back(q2);
    scene.push_back(q3);
    scene.push_back(q4);
    scene.push_back(q5);
    scene.push_back(q6);
}

void my_cornell_box_old() {
    lambertian red(color(0.65f, 0.05f, 0.05f));
    lambertian white(color(0.73f, 0.73f, 0.73f));
    lambertian green(color(0.12f, 0.45f, 0.15f));
    light_material ceiling(color(100.f, 100.f, 100.f));
    materials.push_back(red);
    materials.push_back(white);
    materials.push_back(green);
    materials.push_back(ceiling);

    quad q1(point4(-2.f, -2.f, 2.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q1.mat_idx = 2;
    scene.push_back(q1);

    quad q2(point4(2.f, -2.f, 2.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q2.mat_idx = 0;
    scene.push_back(q2);

    quad q3(point4(-2.f, -2.f, -2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q3.mat_idx = 1;
    scene.push_back(q3);

    quad q4(point4(-2.f, -2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f));
    q4.mat_idx = 1;
    scene.push_back(q4);

    quad q5(point4(-2.f, 2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f));
    q5.mat_idx = 1;
    scene.push_back(q5);

    quad q6(point4(-2.f, -2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q6.mat_idx = 1;
    scene.push_back(q6);

    /*quad area_light(point4(-1.f, 1.99f, 1.f, 0.f), vec4(1.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -1.f, 0.f));
    area_light.mat_idx = 3;
    scene.push_back(area_light);*/

    nsphere ns_light;
    ns_light.mat_idx = 3;
    ns_light.radius = 0.5f;
    ns_light.translate(vec4(0.f, 2.f, -1.f, 0.f));
    scene.push_back(ns_light);

    lambertian lamb(color(1.f, 1.f, 1.f));
    materials.push_back(lamb);
    specular spec(color(1.f, 1.f, 1.f));
    materials.push_back(spec);
    smooth_dielectric glass(1.5f);
    materials.push_back(glass);

    ncube nc;
    nc.mat_idx = 6;
    // point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    // nc->corner = cor;
    scene.push_back(nc);

    nsphere ns2;
    ns2.radius = 0.7f;
    ns2.mat_idx = 5;
    ns2.translate(vec4(-1.f, -1.f, -1.f, 0.f));
    scene.push_back(ns2);

    nsphere ns3;
    ns3.radius = 0.7f;
    ns3.mat_idx = 6;
    ns3.translate(vec4(1.f, -1.f, -1.f, 0.f));
    scene.push_back(ns3);

}

void my_cornell_box() {
    lambertian red(color(0.65f, 0.05f, 0.05f));
    red.in_plane = true;
    lambertian white(color(0.73f, 0.73f, 0.73f));
    white.in_plane = true;
    lambertian green(color(0.12f, 0.45f, 0.15f));
    green.in_plane = true;
    light_material ceiling(color(30.f, 30.f, 30.f));
    materials.push_back(red);
    materials.push_back(white);
    materials.push_back(green);
    materials.push_back(ceiling);

    quad q1(point4(-2.f, -2.f, 2.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q1.mat_idx = 2;
    scene.push_back(q1);

    quad q2(point4(2.f, -2.f, 2.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q2.mat_idx = 0;
    scene.push_back(q2);

    quad q3(point4(-2.f, -2.f, -2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q3.mat_idx = 1;
    scene.push_back(q3);

    quad q4(point4(-2.f, -2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f));
    q4.mat_idx = 1;
    scene.push_back(q4);

    quad q5(point4(-2.f, 2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f));
    q5.mat_idx = 1;
    scene.push_back(q5);

    quad q6(point4(-2.f, -2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q6.mat_idx = 1;
    scene.push_back(q6);

    quad area_light(point4(-1.f, 1.99f, 1.f, 0.f), vec4(1.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -1.f, 0.f));
    area_light.mat_idx = 3;
    //scene.push_back(area_light);

    lambertian lamb(color(1.f, 1.f, 1.f));
    materials.push_back(lamb);
    specular spec(color(1.f, 1.f, 1.f));
    materials.push_back(spec);
    smooth_dielectric glass(1.5f);
    materials.push_back(glass);
    rough_dielectric stained_glass(1.5f);
    stained_glass.alpha = 0.8f;
    // materials.push_back(stained_glass);
    rough_specular metal(color(0.97, 0.74, 0.62));
    metal.alpha = 0.1f;
    // materials.push_back(metal);

    ncube nc;
    nc.half_len = 0.35f;
    nc.mat_idx = 6;
    // point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    // nc->corner = cor;
    scene.push_back(nc);

    nsphere ns2;
    ns2.radius = 0.7f;
    ns2.mat_idx = 4;
    ns2.translate(vec4(-1.f, -1.f, -1.f, 0.f));
    scene.push_back(ns2);

    nsphere ns3;
    ns3.radius = 0.7f;
    ns3.mat_idx = 5;
    ns3.translate(vec4(1.f, -1.f, -1.f, 0.f));
    scene.push_back(ns3);

    ncube nc2;
    nc2.half_len = 0.35f;
    nc2.mat_idx = 6;
    nc2.translate(vec4(1.f, -1.5f, 1.5f, 0.f));
    //scene.push_back(nc2);

    cube c3;
    c3.mat_idx = 2;
    c3.translate(vec4(1.f, 0.f, -1.f, 0.f));
    scene.push_back(c3);
    
    light_material ball(color(4.f, 4.f, 4.f));
    materials.push_back(ball);

    nsphere ns_light;
    ns_light.mat_idx = materials.size() - 1;
    ns_light.radius = 0.5f;
    ns_light.translate(vec4(-1.f, 0.f, 0.f, 0.f));
    scene.push_back(ns_light);

    light_material ball2(color(0.f, 4.f, 4.f));
    materials.push_back(ball2);

    sphere s_light;
    s_light.mat_idx = materials.size() - 1;
    s_light.radius = 0.3f;
    s_light.translate(vec4(1.f, 0.f, 0.f, 0.f));
    scene.push_back(s_light);
}

void my_cornell_box2() {
    lambertian red(color(0.65f, 0.05f, 0.05f));
    lambertian white(color(0.73f, 0.73f, 0.73f));
    lambertian green(color(0.12f, 0.45f, 0.15f));
    light_material ceiling(color(100.f, 100.f, 100.f));
    materials.push_back(red);
    materials.push_back(white);
    materials.push_back(green);
    materials.push_back(ceiling);

    quad q1(point4(-2.f, -2.f, 2.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q1.mat_idx = 2;
    scene.push_back(q1);

    quad q2(point4(2.f, -2.f, 2.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q2.mat_idx = 0;
    scene.push_back(q2);

    quad q3(point4(-2.f, -2.f, -2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q3.mat_idx = 1;
    scene.push_back(q3);

    quad q4(point4(-2.f, -2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f));
    q4.mat_idx = 1;
    scene.push_back(q4);

    quad q5(point4(-2.f, 2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f));
    q5.mat_idx = 1;
    scene.push_back(q5);

    quad q6(point4(-2.f, -2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q6.mat_idx = 1;
    scene.push_back(q6);

    quad area_light(point4(-1.f, 1.99f, 1.f, 0.f), vec4(1.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -1.f, 0.f));
    area_light.mat_idx = 3;
    //scene.push_back(area_light);

    lambertian lamb(color(1.f, 1.f, 1.f));
    materials.push_back(lamb);
    specular spec(color(1.f, 1.f, 1.f));
    materials.push_back(spec);
    smooth_dielectric glass(1.5f);
    materials.push_back(glass);
    rough_dielectric stained_glass(1.5f);
    stained_glass.alpha = 0.8f;
    // materials.push_back(stained_glass);
    rough_specular metal(color(0.97, 0.74, 0.62));
    metal.alpha = 0.1f;
    // materials.push_back(metal);

    ncube nc;
    nc.half_len = 0.35f;
    nc.mat_idx = 6;
    // point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    // nc->corner = cor;
    scene.push_back(nc);

    nsphere ns2;
    ns2.radius = 0.7f;
    ns2.mat_idx = 4;
    ns2.translate(vec4(-1.f, -1.f, -1.f, 0.f));
    scene.push_back(ns2);

    nsphere ns3;
    ns3.radius = 0.7f;
    ns3.mat_idx = 5;
    ns3.translate(vec4(1.f, -1.f, -1.f, 0.f));
    scene.push_back(ns3);

    ncube nc2;
    nc2.half_len = 0.35f;
    nc2.mat_idx = 6;
    nc2.translate(vec4(1.f, -1.5f, 1.5f, 0.f));
    //scene.push_back(nc2);
    
    light_material ball(color(4.f, 4.f, 4.f));
    materials.push_back(ball);

    nsphere ns_light;
    ns_light.mat_idx = 3;
    ns_light.radius = 0.5f;
    ns_light.translate(vec4(0.f, 2.f, -1.f, 0.f));
    scene.push_back(ns_light);

    cube c3;
    c3.mat_idx = 2;  //materials.size() - 1; //2;
    c3.translate(vec4(1.f, 0.f, -1.f, 0.f));
    scene.push_back(c3);

    light_material ball2(color(0.f, 20.f, 20.f));
    materials.push_back(ball2);

    sphere s_light;
    s_light.mat_idx = materials.size() - 1;
    s_light.radius = 0.3f;
    s_light.half_w = 0.01f;
    s_light.translate(vec4(1.f, 0.f, 0.f, 0.f));
    scene.push_back(s_light);
}

void my_cornell_box_white() {
    lambertian red(color(1.f, 1.f, 1.f));
    red.in_plane = true;
    lambertian white(color(1.f, 1.f, 1.f));
    white.in_plane = true;
    lambertian green(color(1.f, 1.f, 1.f));
    green.in_plane = true;
    light_material ceiling(color(1.f, 1.f, 1.f));
    materials.push_back(red);
    materials.push_back(white);
    materials.push_back(green);
    materials.push_back(ceiling);

    quad q1(point4(-2.f, -2.f, 2.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q1.mat_idx = 2;
    scene.push_back(q1);

    quad q2(point4(2.f, -2.f, 2.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q2.mat_idx = 0;
    scene.push_back(q2);

    quad q3(point4(-2.f, -2.f, -2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q3.mat_idx = 1;
    scene.push_back(q3);

    quad q4(point4(-2.f, -2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f));
    q4.mat_idx = 1;
    scene.push_back(q4);

    quad q5(point4(-2.f, 2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -4.f, 0.f));
    q5.mat_idx = 1;
    scene.push_back(q5);

    quad q6(point4(-2.f, -2.f, 2.f, 0.f), vec4(4.f, 0.f, 0.f, 0.f), vec4(0.f, 4.f, 0.f, 0.f));
    q6.mat_idx = 1;
    scene.push_back(q6);

    quad area_light(point4(-1.f, 1.99f, 1.f, 0.f), vec4(1.f, 0.f, 0.f, 0.f), vec4(0.f, 0.f, -1.f, 0.f));
    area_light.mat_idx = 3;
    scene.push_back(area_light);

    lambertian lamb(color(1.f, 1.f, 1.f));
    materials.push_back(lamb);
    specular spec(color(1.f, 1.f, 1.f));
    materials.push_back(spec);
    smooth_dielectric glass(1.5f);
    materials.push_back(glass);
    rough_dielectric stained_glass(1.5f);
    stained_glass.alpha = 0.8f;
    // materials.push_back(stained_glass);
    rough_specular metal(color(0.97, 0.74, 0.62));
    metal.alpha = 0.1f;
    // materials.push_back(metal);

    ncube nc;
    nc.half_len = 0.35f;
    nc.mat_idx = 6;
    // point4 cor(0.25f, 0.25f, 0.25f, 0.25f);
    // nc->corner = cor;
    scene.push_back(nc);

    nsphere ns2;
    ns2.radius = 0.7f;
    ns2.mat_idx = 4;
    ns2.translate(vec4(-1.f, -1.f, -1.f, 0.f));
    scene.push_back(ns2);

    nsphere ns3;
    ns3.radius = 0.7f;
    ns3.mat_idx = 5;
    ns3.translate(vec4(1.f, -1.f, -1.f, 0.f));
    scene.push_back(ns3);

    ncube nc2;
    nc2.half_len = 0.35f;
    nc2.mat_idx = 6;
    nc2.translate(vec4(1.f, -1.5f, 1.5f, 0.f));
    scene.push_back(nc2);

    cube c3;
    c3.mat_idx = 2;
    c3.translate(vec4(1.f, 0.f, -1.f, 0.f));
    scene.push_back(c3);
}

void DI_test() {
    light_material lig(color(1.f, 1.f, 1.f));
    materials.push_back(lig);

    lambertian lamb(color(1.f, 1.f, 1.f));
    materials.push_back(lamb);

    nsphere ns_light;
    ns_light.mat_idx = 0;
    ns_light.radius = 0.5f;
    ns_light.translate(vec4(0.f, 0.f, -1.f, 0.f));
    scene.push_back(ns_light);

    nsphere ns;
    ns.mat_idx = 1;
    ns.radius = 0.5f;
    ns.translate(vec4(1.f, 0.f, -1.f, 0.f));
    scene.push_back(ns);
}

void GI_test() {
    lambertian lamb(color(1.f, 1.f, 1.f));
    materials.push_back(lamb);
    light_material lig(color(0.7f, 0.7f, 0.7f));
    materials.push_back(lig);

    nsphere ns1;
    ns1.mat_idx = 0;
    ns1.radius = 0.5f;
    ns1.translate(vec4(-0.5f, 0.f, 0.f, 0.f));
    scene.push_back(ns1);

    smooth_dielectric glass(1.5f);
    materials.push_back(glass);

    nsphere ns2;
    ns2.mat_idx = 0;
    ns2.radius = 0.5f;
    ns2.translate(vec4(0.5f, 0.f, 0.f, 0.f));
    scene.push_back(ns2);

    nsphere ns3;
    ns3.mat_idx = 0;
    ns3.radius = 0.5f;
    ns3.translate(vec4(1.f, 0.f, 0.f, 0.f));
    //scene.push_back(ns3);

    nsphere env; // constant environment map
    env.mat_idx = 1;
    env.radius = 0.5f;
    scene.push_back(env);
}

void touch_test() {
    lambertian lamb(color(1.f, 1.f, 1.f));
    materials.push_back(lamb);
    light_material lig(color(0.7f, 0.7f, 0.7f));
    materials.push_back(lig);

    light_material env_mat(color(0.f, 0.f, 0.f));
    materials.push_back(env_mat);

    nsphere ns1;
    ns1.mat_idx = 0;
    ns1.radius = 0.5f;
    ns1.translate(vec4(-1.f, 0.f, 0.f, 0.f));
    scene.push_back(ns1);

    nsphere ns2;
    ns2.mat_idx = 1;
    ns2.radius = 0.5f;
    // ns2.translate(vec4(0.f, 0.f, 0.f, 0.f));
    scene.push_back(ns2);

    nsphere env; // constant environment map
    env.mat_idx = 2;
    env.radius = 2.f;
    //scene.push_back(env);
}

void shape_axes() {
    projected_cylinder x_axis;
    x_axis.end0 = point4(AXIS_LEN, 0.f, 0.f, 0.f);
    x_axis.radius = AXIS_RADIUS;
    lambertian x_mat(color(1.f, 0.f, 0.f));
    x_axis.mat_idx = materials.size();
    materials.push_back(x_mat);
    scene.push_back(x_axis);

    projected_cylinder y_axis;
    y_axis.end0 = point4(0.f, AXIS_LEN, 0.f, 0.f);
    y_axis.radius = AXIS_RADIUS;
    lambertian y_mat(color(0.f, 1.f, 0.f));
    y_axis.mat_idx = materials.size();
    materials.push_back(y_mat);
    scene.push_back(y_axis);

    projected_cylinder z_axis;
    z_axis.end0 = point4(0.f, 0.f, AXIS_LEN, 0.f);
    z_axis.radius = AXIS_RADIUS;
    lambertian z_mat(color(0.f, 0.f, 1.f));
    z_axis.mat_idx = materials.size();
    materials.push_back(z_mat);
    scene.push_back(z_axis);

    projected_cylinder w_axis;
    w_axis.end0 = point4(0.f, 0.f, 0.f, AXIS_LEN);
    w_axis.radius = AXIS_RADIUS;
    lambertian w_mat(color(0.73f, 0.33f, 0.827f));
    w_axis.mat_idx = materials.size();
    materials.push_back(w_mat);
    scene.push_back(w_axis);
}

int main() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    g_sm_count = deviceProp.multiProcessorCount;
    g_sm_max_threads = deviceProp.maxThreadsPerMultiProcessor;
    uint stride_threads_per_block = 256;
    uint stride_num_blocks = (g_sm_max_threads / stride_threads_per_block) * g_sm_count;
    printf("SM count %d | Max threads per SM %d | Stride block count %d\n", g_sm_count, g_sm_max_threads, stride_num_blocks);

    math_test<<<1, 1>>>();

    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
        return 1;
    }

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor()); // Valid on GLFW 3.3+ only
    GLFWwindow* window = glfwCreateWindow((int)(1280 * main_scale), (int)(800 * main_scale), "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

     // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    style.FontScaleDpi = main_scale;        // Set initial font scale. (using io.ConfigDpiScaleFonts=true makes this unnecessary. We leave both here for documentation purpose)

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    //spheres();
    //quad_light_test();
    //my_cornell_box_old();
    //my_cornell_box();
    //my_cornell_box2();
    // DI_test();
    GI_test();
    //touch_test();
    //my_cornell_box_white();
    //tesseract_lines();
    //tesseract();
    //tesseract_reflector();
    //tesseract_lines_reflector();
    //tesseract_glass();
    //all_white();
    //shape_axes();
    
    size_t scene_len = scene.size();
    gpuErrchk(cudaMemcpyToSymbol(d_scene_len, &scene_len, sizeof(size_t)));
    size_t lights_len = lights.size();
    gpuErrchk(cudaMemcpyToSymbol(d_lights_len, &lights_len, sizeof(size_t)));    
    size_t materials_len = materials.size();
    gpuErrchk(cudaMemcpyToSymbol(d_materials_len, &materials_len, sizeof(size_t)));

    // can try space distortion too
    // tesseract lines with a sphere in the middle?
    // tesseract solid reflectors?

    camera::aspect_ratio = 16.f / 9.f;
    camera::image_width = 1280;
    // camera::render_normals = true;
    camera::initialize();

    // --- SETUP OPENGL TEXTURE ---
    GLuint render_texture;
    glGenTextures(1, &render_texture);
    glBindTexture(GL_TEXTURE_2D, render_texture);
    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Allocate memory for the texture on the GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, camera::image_width, camera::image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind

    gpuErrchk(cudaGraphicsGLRegisterImage(
        &render_texture_CUDA,
        render_texture,
        GL_TEXTURE_2D, // Target of the texture (e.g., GL_TEXTURE_2D)
        cudaGraphicsMapFlagsWriteDiscard
    ));

    float prev_frame = 0.f;
    float current_frame = 0.f;

    std::vector<std::thread> threads(NUM_CPU_THREADS);
    bool is_rendering = true; //std::atomic<bool> is_rendering = true;
    uint row_range = std::ceil((float)camera::image_height / NUM_CPU_THREADS); // range: ceil(height / N)

    uint num_pixels = camera::image_width * camera::image_height;
    color* d_color_buffer;
    gpuErrchk(cudaMalloc(&d_color_buffer, num_pixels * sizeof(color)));

    uchar4* d_image_data;
    gpuErrchk(cudaMalloc(&d_image_data, num_pixels * sizeof(uchar4)));

    //nvtxRangePush("Processing Inputs");
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((camera::image_height + threads_per_block.x - 1) / threads_per_block.x, 
    (camera::image_width + threads_per_block.y - 1) / threads_per_block.y);

    uint64_t* d_pcg_states;
    gpuErrchk(cudaMalloc(&d_pcg_states, num_pixels * sizeof(uint64_t)));
    camera::init_pcg_states_kernel<<<num_blocks, threads_per_block>>>(d_pcg_states, camera::image_width, camera::image_height);

    int num_samples = 0;

    while (!glfwWindowShouldClose(window)) {
        current_frame = glfwGetTime();
        fixed_delta_time = current_frame - prev_frame;

        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        if (fixed_delta_time < MIN_DELTA_TIME) {
            continue;
        }
        prev_frame = current_frame;

        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (false) {
            test_render_kernel<<<num_blocks, threads_per_block>>>(d_image_data, camera::image_width, camera::image_height, static_cast<float>(glfwGetTime()));

            cudaArray* d_texture_array = nullptr; // Pointer to the CUDA array representing the texture
            gpuErrchk(cudaGraphicsMapResources(1, &render_texture_CUDA, 0)); // Map on stream 0
            gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&d_texture_array, render_texture_CUDA, 0, 0));

            gpuErrchk(cudaMemcpy2DToArray(d_texture_array, // Destination: CUDA array
                    0, 0,             // Destination X, Y offsets (start at top-left)
                    d_image_data,      // Source: Device pointer to your linear pixel data
                    camera::image_width * sizeof(uchar4), // Source pitch (bytes per row)
                    camera::image_width * sizeof(uchar4), // Width of the copy (bytes)
                    camera::image_height, // Height of the copy (rows)
                    cudaMemcpyDeviceToDevice)); // Type of copy (Device to Array)
            gpuErrchk(cudaGraphicsUnmapResources(1, &render_texture_CUDA, 0));
            //glBindTexture(GL_TEXTURE_2D, render_texture);
            //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camera::image_width, camera::image_height, GL_RGB, GL_UNSIGNED_BYTE, camera::image_data.data());
            //glBindTexture(GL_TEXTURE_2D, 0); // Unbind
        }

                // --- Render Target Window ---
        ImGui::Begin("Render Output");
        //bool input_changed = handle_inputs();
        bool input_changed_cuda = handle_inputs_cuda();
        if (input_changed_cuda) {
            num_samples = 0;
            gpuErrchk(cudaMemset(d_color_buffer, 0, num_pixels * sizeof(color)));
            gpuErrchk(cudaMemset(d_image_data, 0, num_pixels * sizeof(uchar4)));
        }

        if (true) {
            camera::render_kernel<<<num_blocks, threads_per_block, scene.data_size + lights.data_size + materials.data_size + scene.size() * sizeof(shape*) + lights.size() * sizeof(light*) + materials.size() * sizeof(material*)>>>
            (num_samples, false, d_color_buffer, d_image_data, d_pcg_states, camera::image_width, camera::image_height, scene.d_data, scene.data_size, lights.d_data, lights.data_size, materials.d_data, materials.data_size);
            num_samples++;

            //camera::render_stride_kernel<<<stride_num_blocks, stride_threads_per_block>>>(d_image_data, camera::image_width, camera::image_height * camera::image_width);

            cudaArray* d_texture_array = nullptr; // Pointer to the CUDA array representing the texture
            gpuErrchk(cudaGraphicsMapResources(1, &render_texture_CUDA, 0)); // Map on stream 0
            gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&d_texture_array, render_texture_CUDA, 0, 0));

            gpuErrchk(cudaMemcpy2DToArray(d_texture_array, // Destination: CUDA array
                    0, 0,             // Destination X, Y offsets (start at top-left)
                    d_image_data,      // Source: Device pointer to your linear pixel data
                    camera::image_width * sizeof(uchar4), // Source pitch (bytes per row)
                    camera::image_width * sizeof(uchar4), // Width of the copy (bytes)
                    camera::image_height, // Height of the copy (rows)
                    cudaMemcpyDeviceToDevice)); // Type of copy (Device to Array)
            gpuErrchk(cudaGraphicsUnmapResources(1, &render_texture_CUDA, 0));
            //nvtxRangePop();
        }

        if (false) {
            uint first_row = 0;

            for (uint i = 0; i < NUM_CPU_THREADS; i++) {
                uint last_row = std::min(first_row + row_range, camera::image_height - 1);

                threads[i] = std::thread([first_row, last_row]() { 
                    camera::render_rt(first_row, last_row);
                });

                first_row += row_range;
            }

            for (uint i = 0; i < NUM_CPU_THREADS; i++) {
                threads[i].join();
            }

            //glBindTexture(GL_TEXTURE_2D, render_texture);
            //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camera::image_width, camera::image_height, GL_RGBA, GL_UNSIGNED_BYTE, image_data.data());
            //glBindTexture(GL_TEXTURE_2D, 0); // Unbind
        }

        // Display the texture in an ImGui::Image widget
        ImGui::Image((void*)(intptr_t)render_texture, ImVec2(camera::image_width, camera::image_height));
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        
        for (size_t i = 0; i < materials_len; i++) {
            std::string label = "material " + std::to_string(i);

            if (ImGui::Button(label.c_str())) {
                world::toggle_planar_reflection_kernel<<<1,1>>>(i, materials.d_list);
                num_samples = 0;
                gpuErrchk(cudaMemset(d_color_buffer, 0, num_pixels * sizeof(color)));
                gpuErrchk(cudaMemset(d_image_data, 0, num_pixels * sizeof(uchar4)));
            }

            ImGui::SameLine();
        }

        ImGui::NewLine();

        if (ImGui::Button("EXPORT IMAGE")) {
            camera::export_image(d_image_data, num_pixels);
        }
        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    delete &scene;
    delete &lights;
    delete &materials;
    gpuErrchk(cudaGraphicsUnregisterResource(render_texture_CUDA));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(d_color_buffer));
    gpuErrchk(cudaFree(d_image_data));
    gpuErrchk(cudaFree(d_pcg_states));

    is_rendering = false;
    glDeleteTextures(1, &render_texture); // Clean up the texture
    
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    // scene1();
    // scene2();
    // math_test();
    // scene3();
    // rotation_test();
	return 0;
}