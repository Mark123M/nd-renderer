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
#include "ncube.h"
#include "cylinder.h"
#include "camera.h"
#include "world.h"
#include "direction_light.h"
#include "math_util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

struct cudaGraphicsResource* render_texture_CUDA = nullptr;

__global__ void test_render_kernel(uchar4* d_image_data, int width, int height, float time)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

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

static constexpr int num_cpu_threads = 20;

static bool handle_inputs_cuda() {
    bool did_input = false;
    uint threads_per_block = 64;
    uint num_blocks = (scene.size() + threads_per_block - 1) / threads_per_block;

    if (ImGui::IsKeyPressed(ImGuiKey_A)) {
        world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(-MOVE_AMOUNT, 0.f, 0.f, 0.f));
        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_D)) {
        world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(MOVE_AMOUNT, 0.f, 0.f, 0.f));
        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_W)) {
        world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(0.f, MOVE_AMOUNT, 0.f, 0.f));
        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_S)) {
        world::translate_kernel<<<num_blocks, threads_per_block>>>(vec4(0.f, -MOVE_AMOUNT, 0.f, 0.f));
        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        world::rotate_xz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_AMOUNT);
        world::rotate_yw_kernel<<<num_blocks, threads_per_block>>>(ROTATE_AMOUNT);
        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
        world::rotate_yz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_AMOUNT);
        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        world::rotate_xz_kernel<<<num_blocks, threads_per_block>>>(ROTATE_AMOUNT);
        did_input = true;
    }

    return did_input;
}

static float handle_inputs() {
    constexpr float move_amount = 0.03f;
    constexpr float rotate_amount = 0.05f;

    bool did_input = false;

    if (ImGui::IsKeyPressed(ImGuiKey_A)) {
        for (shape* c : scene) {
            c->basis.translate(vec4(-move_amount, 0.f, 0.f, 0.f));
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_D)) {
        for (shape* c : scene) {
            c->basis.translate(vec4(move_amount, 0.f, 0.f, 0.f));
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_W)) {
        for (shape* c : scene) {
            c->basis.translate(vec4(0.f, move_amount, 0.f, 0.f));
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_S)) {
        for (shape* c : scene) {
            c->basis.translate(vec4(0.f, -move_amount, 0.f, 0.f));
        }
        
        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        for (shape* c : scene) {
            c->basis.rotate_xz(rotate_amount);
            c->basis.rotate_yw(rotate_amount);
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
        for (shape* c : scene) {
            c->basis.rotate_yz(rotate_amount);
        }

        did_input = true;
    }

    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) {
        for (shape* c : scene) {
            c->basis.rotate_xz(rotate_amount);
        }

        did_input = true;
    }

    return did_input;
}

int main() {
    hello<<<1, 64>>>();

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
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
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

    int edges[32][2] = {
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
    std::vector<std::unique_ptr<cylinder>> unique_scene;

    for (int i = 0; i < 32; ++i) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        vec4 move(L / 2, L / 2, L / 2, L / 2);
        unique_scene.push_back(std::make_unique<cylinder>());
        unique_scene.back()->a = vertices[idx1] - move;
        unique_scene.back()->b = vertices[idx2] - move;
        unique_scene.back()->albedo = edge_color;
        scene.push_back(unique_scene.back().get());
    }

    /*std::unique_ptr<nsphere> ball = std::make_unique<nsphere>();
    ball->radius = 0.5f;
    ball->albedo = color(1.f, 0.f, 0.f);
    scene.push_back(ball.get());*/

    std::unique_ptr<cylinder> x_axis = std::make_unique<cylinder>();
    x_axis->b = point4(L, 0, 0, 0);
    x_axis->radius = 0.01f;
    x_axis->albedo = color(1, 0, 0);
    std::unique_ptr<cylinder> y_axis = std::make_unique<cylinder>();
    y_axis->b = point4(0, L, 0, 0);
    y_axis->radius = 0.01f;
    y_axis->albedo = color(0, 1, 0);
    std::unique_ptr<cylinder> z_axis = std::make_unique<cylinder>();
    z_axis->b = point4(0, 0, L, 0);
    z_axis->radius = 0.01f;
    z_axis->albedo = color(0, 0, 1);
    std::unique_ptr<cylinder> w_axis = std::make_unique<cylinder>();
    w_axis->b = point4(0, 0, 0, L);
    w_axis->radius = 0.01f;
    w_axis->albedo = color(0.73f, 0.33f, 0.827f);

    scene.push_back(x_axis.get());
    scene.push_back(y_axis.get());
    scene.push_back(z_axis.get());
    scene.push_back(w_axis.get());

    std::unique_ptr<direction_light> lig1 = std::make_unique<direction_light>();
    lig1->col = color(1.0f, 1.0f, 0.9f);
    lig1->dir = vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f));
    lights.push_back(lig1.get());

    world::initialize();
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

    int rt_latency = 0;
    int full_latency = 0;

    std::vector<std::thread> threads(num_cpu_threads);
    bool is_rendering = true; //std::atomic<bool> is_rendering = true;
    int row_range = std::ceil((float)camera::image_height / num_cpu_threads); // range: ceil(height / N)

    image_data = std::vector<uchar4>(camera::image_width * camera::image_height);
    uchar4* d_image_data;
    int numel = camera::image_width * camera::image_height;
    gpuErrchk(cudaMalloc(&d_image_data, numel * sizeof(uchar4)));

    while (!glfwWindowShouldClose(window)) {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (false) {
            dim3 threads_per_block(16, 16);
            dim3 num_blocks((camera::image_height + threads_per_block.x - 1) / threads_per_block.x, 
            (camera::image_width + threads_per_block.y - 1) / threads_per_block.y);
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
            dim3 threads_per_block(16, 16);
            dim3 num_blocks((camera::image_height + threads_per_block.x - 1) / threads_per_block.x, 
            (camera::image_width + threads_per_block.y - 1) / threads_per_block.y);
            camera::render_kernel<<<num_blocks, threads_per_block>>>(d_image_data, camera::image_width, camera::image_height);

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
        }

        if (false) {
            int first_row = 0;

            for (int i = 0; i < num_cpu_threads; i++) {
                int last_row = std::min(first_row + row_range, camera::image_height - 1);

                threads[i] = std::thread([first_row, last_row]() { 
                    camera::render_rt(first_row, last_row);
                });

                first_row += row_range;
            }

            for (int i = 0; i < num_cpu_threads; i++) {
                threads[i].join();
            }

            glBindTexture(GL_TEXTURE_2D, render_texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camera::image_width, camera::image_height, GL_RGBA, GL_UNSIGNED_BYTE, image_data.data());
            glBindTexture(GL_TEXTURE_2D, 0); // Unbind
        }

        // Display the texture in an ImGui::Image widget
        ImGui::Image((void*)(intptr_t)render_texture, ImVec2(camera::image_width, camera::image_height));
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::Text("Raytracer latency: %dms   Transfer latency: %dms", rt_latency, full_latency - rt_latency);
        //ImGui::Text("Tesseract Center: (%.3f, %.3f, %.3f, %.3f)", center.x, center.y, center.z, center.w);
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

    world::destruct<<<1,1>>>();
    gpuErrchk(cudaGraphicsUnregisterResource(render_texture_CUDA));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaFree(d_image_data));

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