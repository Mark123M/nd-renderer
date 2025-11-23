#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <thread>
#include <fstream>
#include <filesystem>
#include "json/include/nlohmann/json.hpp"
using json = nlohmann::json;

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
#include "bvh.h"

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

struct input_result {
    bool should_clear_buffer = false;
    bool should_rebuild_bvh = false;
};

static input_result handle_inputs_cuda(std::vector<shape_wrapper>& shapes, device_list<shape>& scene) {
    input_result res;
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

        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_Z)) {
        if (mouse_delta.x != 0.f) {
            world::rotate_camera_yw_kernel<<<1, 1>>>(mouse_delta.x * CAMERA_ROTATE_RATE * fixed_delta_time);
        }

        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_C)) {
        if (mouse_delta.x != 0.f) {
            world::rotate_camera_xw_kernel<<<1, 1>>>(mouse_delta.x * CAMERA_ROTATE_RATE * fixed_delta_time);
        }

        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_A)) {
        //world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(-MOVE_RATE * fixed_delta_time, 0.f, 0.f, 0.f));
        world::move_camera_x_kernel<<<1, 1>>>(-CAMERA_MOVE_RATE * fixed_delta_time);
        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_D)) {
        //world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(MOVE_RATE * fixed_delta_time, 0.f, 0.f, 0.f));
        world::move_camera_x_kernel<<<1, 1>>>(CAMERA_MOVE_RATE * fixed_delta_time);
        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_W)) {
        //world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(0.f, MOVE_RATE * fixed_delta_time, 0.f, 0.f));
        world::move_camera_z_kernel<<<1, 1>>>(-CAMERA_MOVE_RATE * fixed_delta_time);
        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_S)) {
        //world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(0.f, -MOVE_RATE * fixed_delta_time, 0.f, 0.f));
        world::move_camera_z_kernel<<<1, 1>>>(CAMERA_MOVE_RATE * fixed_delta_time);
        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_Q)) {
        world::move_camera_w_kernel<<<1, 1>>>(CAMERA_MOVE_RATE * fixed_delta_time);
        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_E)) {
        world::move_camera_w_kernel<<<1, 1>>>(-CAMERA_MOVE_RATE * fixed_delta_time);
        res.should_clear_buffer = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) {
        for (shape_wrapper& sw : shapes) {
            sw.s->rotate_xz(ROTATE_RATE * fixed_delta_time);
            sw.s->rotate_yw(ROTATE_RATE * fixed_delta_time);
        }

        world::rotate_xz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_RATE * fixed_delta_time, scene.d_list);
        world::rotate_yw_kernel<<<num_blocks, threads_per_block>>>(ROTATE_RATE * fixed_delta_time, scene.d_list);
        res.should_clear_buffer = true;
        res.should_rebuild_bvh = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) {
        for (shape_wrapper& sw : shapes) {
            sw.s->rotate_yz(ROTATE_RATE * fixed_delta_time);
        }

        world::rotate_yz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_RATE * fixed_delta_time, scene.d_list);
        res.should_clear_buffer = true;
        res.should_rebuild_bvh = true;
    }

    if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) {
        for (shape_wrapper& sw : shapes) {
            sw.s->rotate_xz(ROTATE_RATE * fixed_delta_time);
        }

        world::rotate_xz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_RATE * fixed_delta_time, scene.d_list);
        res.should_clear_buffer = true;
        res.should_rebuild_bvh = true;
    }

    return res;
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

void bvh_test() {
    std::cout << "==========[BVH TEST (ONE SHAPE)]==========" << std::endl;
    std::vector<shape_wrapper> shapes;
    nsphere ns1;
    ns1.radius = 0.5f;
    ns1.translate(vec4(-1.f, 0.f, 0.f, 0.f));
    shapes.emplace_back(std::string("nsphere"), ns1);

    uint num_nodes = 0;
    bvh_node* root = build_recursive(0, shapes.size() - 1, num_nodes, shapes);
    log_bvh(root, 0);
    delete root;

    std::cout << "==========[BVH TEST (TWO SHAPES)]==========" << std::endl;
    nsphere ns2;
    ns2.radius = 0.5f;
    ns2.translate(vec4(1.f, 0.f, 0.f, 0.f));
    shapes.emplace_back(std::string("nsphere"), ns2);

    num_nodes = 0;
    root = build_recursive(0, shapes.size() - 1, num_nodes, shapes);
    log_bvh(root, 0);
    delete root;

    std::cout << "==========[BVH TEST (FOUR SHAPES)]==========" << std::endl;
    nsphere ns4;
    ns4.radius = 0.5f;
    ns4.translate(vec4(4.f, 0.f, 0.f, 0.f));
    shapes.emplace_back(std::string("nsphere"), ns4);

    nsphere ns3;
    ns3.radius = 0.5f;
    ns3.translate(vec4(-4.f, 0.f, 0.f, 0.f));
    shapes.emplace_back(std::string("nsphere"), ns3);

    num_nodes = 0;
    root = build_recursive(0, shapes.size() - 1, num_nodes, shapes);
    log_bvh(root, 0);
    delete root;

    std::cout << "==========[BVH TEST (MULTI-LEAF)]==========" << std::endl;
    nsphere ns5;
    ns5.radius = 1.f;
    ns5.translate(vec4(-4.f, 0.f, 0.f, 0.f));
    shapes.emplace_back(std::string("nsphere"), ns5);

    /*num_nodes = 0;
    root = build_recursive(0, shapes.size() - 1, num_nodes, shapes);
    log_bvh(root, 0);
    delete root; */
    std::vector<linear_bvh_node> linear_nodes;
    build_bvh(shapes, linear_nodes);
    for (const auto& l : linear_nodes) {
        std::cout << linear_bvh_node::to_string(l) << std::endl;
    }
}

void build_bvh_cuda(std::vector<linear_bvh_node>& linear_nodes, linear_bvh_node*& d_linear_nodes, std::vector<shape_wrapper>& shapes) {
    linear_nodes.clear();
    build_bvh(shapes, linear_nodes);
    for (const auto& l : linear_nodes) {
        std::cout << linear_bvh_node::to_string(l) << std::endl;
    }

    uint linear_nodes_bytes = linear_nodes.size() * sizeof(linear_bvh_node);
    if (d_linear_nodes != nullptr) {
        gpuErrchk(cudaFree(d_linear_nodes));
    }
    gpuErrchk(cudaMalloc(&d_linear_nodes, linear_nodes_bytes));
    linear_bvh_node* h_linear_nodes = linear_nodes.data();
    gpuErrchk(cudaMemcpy(d_linear_nodes, h_linear_nodes, linear_nodes_bytes, cudaMemcpyHostToDevice));
}

void build_base_shape(shape* s, size_t mat_idx, const vec4& t, float rxy, float rxz, float rxw, float ryz, float ryw, float rzw) {
    s->mat_idx = mat_idx;
    s->translate(t);
    s->rotate_xy(rxy);
    s->rotate_xz(rxz);
    s->rotate_xw(rxw);
    s->rotate_yz(ryz);
    s->rotate_yw(ryw);
    s->rotate_zw(rzw);
}

void build_scene(
    json& data,
    std::vector<linear_bvh_node>& linear_nodes,
    linear_bvh_node*& d_linear_nodes,
    std::vector<shape_wrapper>& shapes,
    device_list<shape>& scene,
    device_list<material>& materials,
    device_list<light>& lights
) {
    shapes.clear();
    scene.clear();
    materials.clear();
    lights.clear();

    for (auto& shape_data : data["shapes"]) {
        std::string shape_class = shape_data["class"].template get<std::string>();
        size_t mat_idx = shape_data["mat_idx"].template get<size_t>();
        vec4 t(0.f, 0.f, 0.f, 0.f);
        float rxy = 0.f;
        float rxz = 0.f;
        float rxw = 0.f;
        float ryz = 0.f;
        float ryw = 0.f; 
        float rzw = 0.f;
        // vec4 r(0.f, 0.f, 0.f, 0.f);
        if (shape_data.find("translate") != shape_data.end()) {
            t = shape_data["translate"].template get<vec4>();
        }

        if (shape_data.find("rotate_xy") != shape_data.end()) {
            rxy = shape_data["rotate_xy"].template get<float>();
        }
        if (shape_data.find("rotate_xz") != shape_data.end()) {
            rxz = shape_data["rotate_xz"].template get<float>();
        }
        if (shape_data.find("rotate_xw") != shape_data.end()) {
            rxw = shape_data["rotate_xw"].template get<float>();
        }
        if (shape_data.find("rotate_yz") != shape_data.end()) {
            ryz = shape_data["rotate_yz"].template get<float>();
        }
        if (shape_data.find("rotate_yw") != shape_data.end()) {
            ryw = shape_data["rotate_yw"].template get<float>();
        }
        if (shape_data.find("rotate_zw") != shape_data.end()) {
            rzw = shape_data["rotate_zw"].template get<float>();
        }

        if (shape_class == "nsphere") {
            nsphere ns;

            if (shape_data.find("radius") != shape_data.end()) {
                ns.radius = shape_data["radius"].template get<float>();
            }

            std::cout << "constructing nsphere radius " << ns.get_radius() << " translation " << t.x << " " << t.y << " " << t.z << " " << t.w << std::endl;
            build_base_shape(&ns, mat_idx, t, rxy, rxz, rxw, ryz, ryw, rzw);
            shapes.emplace_back(shape_class, ns);
        } else if (shape_class == "ncube") {
            ncube nc;

            if (shape_data.find("half_len") != shape_data.end()) {
                nc.half_len = shape_data["half_len"].template get<float>();
            }

            std::cout << "constructing ncube half_len " << nc.half_len << " translation " << t.x << " " << t.y << " " << t.z << " " << t.w << std::endl;
            build_base_shape(&nc, mat_idx, t, rxy, rxz, rxw, ryz, ryw, rzw);
            shapes.emplace_back(shape_class, nc);
        } else if (shape_class == "sphere") {
            sphere s;

            if (shape_data.find("radius") != shape_data.end()) {
                s.radius = shape_data["radius"].template get<float>();
            }

            if (shape_data.find("half_w") != shape_data.end()) {
                s.half_w = shape_data["half_w"].template get<float>();
            }

            std::cout << "constructing sphere radius " << s.radius << " half_w " << s.half_w << std::endl;
            build_base_shape(&s, mat_idx, t, rxy, rxz, rxw, ryz, ryw, rzw);
            shapes.emplace_back(shape_class, s);
        } else if (shape_class == "cube") {
            cube c;

            if (shape_data.find("half_len") != shape_data.end()) {
                c.half_len = shape_data["half_len"].template get<float>();
            }

            if (shape_data.find("half_w") != shape_data.end()) {
                c.half_w = shape_data["half_w"].template get<float>();
            }

            std::cout << "constructing cube half_len " << c.half_len << " half_w" << c.half_w << std::endl;
            build_base_shape(&c, mat_idx, t, rxy, rxz, rxw, ryz, ryw, rzw);
            shapes.emplace_back(shape_class, c);
        } else if (shape_class == "quad") {
            point4 o = shape_data["origin"].template get<point4>();
            vec4 u = shape_data["u"].template get<vec4>();
            vec4 v = shape_data["v"].template get<vec4>();
            quad q(o, u, v);

            std::cout << "constructing quad" << std::endl;
            build_base_shape(&q, mat_idx, t, rxy, rxz, rxw, ryz, ryw, rzw);
            shapes.emplace_back(shape_class, q);
        }
    }
    
    build_bvh_cuda(linear_nodes, d_linear_nodes, shapes);

    for (const shape_wrapper& sw : shapes) {
        if (sw.type == "nsphere") {
            scene.push_back<nsphere>(sw.s.get());
        } else if (sw.type == "ncube") {
            scene.push_back<ncube>(sw.s.get());
        } else if (sw.type == "sphere") {
            scene.push_back<sphere>(sw.s.get());
        } else if (sw.type == "cube") {
            scene.push_back<cube>(sw.s.get());
        } else if (sw.type == "quad") {
            scene.push_back<quad>(sw.s.get());
        }
    }

    for (auto& mat_data : data["materials"]) {
        std::string mat_class = mat_data["class"].template get<std::string>();

        if (mat_class == "lambertian") {
            lambertian l;
            
            if (mat_data.find("albedo") != mat_data.end()) {
                color c = mat_data["albedo"].template get<color>();
                l.albedo = c;
            }

            std::cout << "constructing lambertian albedo " << l.albedo.r << " " << l.albedo.g << " " << l.albedo.b << std::endl;
            materials.push_back<lambertian>(l);
        } else if (mat_class == "light_material") {
            light_material lig;
            
            if (mat_data.find("color") != mat_data.end()) {
                color c = mat_data["color"].template get<color>();
                lig.col = c;
            }

            if (mat_data.find("albedo") != mat_data.end()) {
                color c = mat_data["albedo"].template get<color>();
                lig.albedo = c;
            }

            std::cout << "constructing light color " << lig.col.r << " " << lig.col.g << " " << lig.col.b << " albedo " << lig.albedo.r << " " << lig.albedo.g << " " << lig.albedo.b << std::endl;
            materials.push_back<light_material>(lig);
        } else if (mat_class == "dielectric") {
            smooth_dielectric sd;

            if (mat_data.find("eta") != mat_data.end()) {
                sd.eta = mat_data["eta"].template get<float>();
            }

            materials.push_back<smooth_dielectric>(sd);
        } else if (mat_class == "specular") {
            specular sp;

            if (mat_data.find("albedo") != mat_data.end()) {
                sp.albedo = mat_data["albedo"].template get<color>();
            }

            materials.push_back<specular>(sp);
        }
    }

    size_t scene_len = scene.size();
    gpuErrchk(cudaMemcpyToSymbol(d_scene_len, &scene_len, sizeof(size_t)));
    size_t lights_len = lights.size();
    gpuErrchk(cudaMemcpyToSymbol(d_lights_len, &lights_len, sizeof(size_t)));    
    size_t materials_len = materials.size();
    gpuErrchk(cudaMemcpyToSymbol(d_materials_len, &materials_len, sizeof(size_t)));
}

void shape_axes(device_list<shape>& scene, device_list<material>& materials, device_list<light>& lights) {
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
    //my_cornell_box2(scene, materials, lights);
    // DI_test();
    //GI_test();
    //touch_test();
    //my_cornell_box_white();
    //tesseract_lines();
    //tesseract();
    //tesseract_reflector();
    //tesseract_lines_reflector();
    //tesseract_glass();
    //all_white();
    //shape_axes();
    // can try space distortion too
    // tesseract lines with a sphere in the middle?
    // tesseract solid reflectors?
    //bvh_test();
    //return 0;

    std::vector<linear_bvh_node> linear_nodes;
    linear_bvh_node* d_linear_nodes = nullptr;
    std::vector<shape_wrapper> shapes;
    device_list<shape> scene;
    device_list<light> lights;
    device_list<material> materials;

    std::filesystem::path init_scene_path("scenes/cornell2.json");
    std::filesystem::directory_entry cur_scene_entry(init_scene_path);
    std::filesystem::file_time_type cur_scene_last_write_time = cur_scene_entry.last_write_time();
    std::ifstream f(init_scene_path);
    json data = json::parse(f);
    build_scene(data, linear_nodes, d_linear_nodes, shapes, scene, materials, lights);
    //std::cout << data["shapes"][0]["class"] << std::endl;

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
    double3* d_color_buffer;
    gpuErrchk(cudaMalloc(&d_color_buffer, num_pixels * sizeof(double3)));

    uchar4* d_image_data;
    gpuErrchk(cudaMalloc(&d_image_data, num_pixels * sizeof(uchar4)));

    //nvtxRangePush("Processing Inputs");
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((camera::image_height + threads_per_block.x - 1) / threads_per_block.x, 
    (camera::image_width + threads_per_block.y - 1) / threads_per_block.y);

    uint64_t* d_pcg_states;
    gpuErrchk(cudaMalloc(&d_pcg_states, num_pixels * sizeof(uint64_t)));
    camera::init_pcg_states_kernel<<<num_blocks, threads_per_block>>>(d_pcg_states, camera::image_width, camera::image_height);

    int num_samples = 0;
    auto clear_buffer = [&num_samples, num_pixels, d_color_buffer, d_image_data]() -> void {
        num_samples = 0;
        gpuErrchk(cudaMemset(d_color_buffer, 0, num_pixels * sizeof(double3)));
        gpuErrchk(cudaMemset(d_image_data, 0, num_pixels * sizeof(uchar4)));
    };

    int sample_mode = 0;
    bool sample_lights = true; // NEE;

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
        input_result res = handle_inputs_cuda(shapes, scene);
        if (res.should_clear_buffer) {
            clear_buffer();
        }
        if (res.should_rebuild_bvh) {
            build_bvh_cuda(linear_nodes, d_linear_nodes, shapes);
        }

        if (cur_scene_entry.last_write_time() > cur_scene_last_write_time) {
            cur_scene_last_write_time = cur_scene_entry.last_write_time();
            std::ifstream f(cur_scene_entry.path());
            json data = json::parse(f);
            build_scene(data, linear_nodes, d_linear_nodes, shapes, scene, materials, lights);
            clear_buffer();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_R)) {
            clear_buffer();
            world::reset_camera_kernel<<<1,1>>>();
            gpuErrchk(cudaDeviceSynchronize());
        }

        uint linear_nodes_bytes = linear_nodes.size() * sizeof(linear_bvh_node);
        uint shared_memory_bytes = linear_nodes_bytes + scene.data_size + lights.data_size + materials.data_size + scene.size() * sizeof(shape*) + lights.size() * sizeof(light*) + materials.size() * sizeof(material*);
        camera::render_kernel<<<num_blocks, threads_per_block, shared_memory_bytes>>>(
            num_samples,
            sample_mode,
            sample_lights,
            camera::image_width,
            camera::image_height,
            d_linear_nodes,
            linear_nodes_bytes,
            scene.d_data,
            scene.data_size,
            lights.d_data,
            lights.data_size,
            materials.d_data,
            materials.data_size,
            d_color_buffer,
            d_image_data,
            d_pcg_states
        );

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

        // Display the texture in an ImGui::Image widget
        ImGui::Image((void*)(intptr_t)render_texture, ImVec2(camera::image_width, camera::image_height));
        ImGui::Text("%.3f ms/frame (%.1f FPS), %d samples per pixel", 1000.0f / io.Framerate, io.Framerate, num_samples);
        
        for (const auto& entry : std::filesystem::directory_iterator(PATH_TO_SCENES)) {
            const char* stem = entry.path().stem().c_str();
            
            if (ImGui::Button(stem)) {
                cur_scene_entry = entry;
                cur_scene_last_write_time = cur_scene_entry.last_write_time();
                std::ifstream f(entry.path());
                json data = json::parse(f);
                build_scene(data, linear_nodes, d_linear_nodes, shapes, scene, materials, lights);
                clear_buffer();
            }

            ImGui::SameLine();
        }

        ImGui::NewLine();

        for (size_t i = 0; i < materials.size(); i++) {
            std::string label = "material " + std::to_string(i);

            if (ImGui::Button(label.c_str())) {
                world::toggle_planar_reflection_kernel<<<1,1>>>(i, materials.d_list);
                clear_buffer();
            }

            ImGui::SameLine();
        }

        ImGui::NewLine();
        
        if (ImGui::RadioButton("Sample solid angles", &sample_mode, 0)) {
            clear_buffer();
        }

        ImGui::SameLine();

        if (ImGui::RadioButton("Sample surface area", &sample_mode, 1)) {
            clear_buffer();
        }

        if (ImGui::Checkbox("Sample lights", &sample_lights)) {
            clear_buffer();
        }

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

    gpuErrchk(cudaGraphicsUnregisterResource(render_texture_CUDA));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(d_linear_nodes));
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