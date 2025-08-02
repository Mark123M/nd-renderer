#include <iostream>
#include <sstream>
#include <vector>
#include "color.h"
#include "ray.h"
#include "nsphere.h"
#include "ncube.h"
#include "cylinder.h"
#include "camera.h"
#include "direction_light.h"
#include "mat4.h"
#include "util.h"
#include <memory>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <thread>
#include <barrier>

static constexpr int N = 20;

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static void scene1() {
    std::vector<shape*> scene;
    std::unique_ptr<nsphere> sphere1 = std::make_unique<nsphere>(point4(0.f, 0.f, -1.f, 0.4f), 0.5f, color(1.f, 0.647f, 0.f), 4);
    std::unique_ptr<nsphere> sphere2 = std::make_unique<nsphere>(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
    scene.push_back(sphere1.get());
    scene.push_back(sphere2.get());

    std::vector<direction_light> lights;
    direction_light light1(vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.f)), color(1.0f, 1.0f, 0.9f));
    lights.push_back(light1);

    camera cam{ scene, lights };
    cam.aspect_ratio = 16.f / 9.f;
    cam.image_width = 400;
    //cam.render_normals = true;
    cam.render();

    for (shape* shape_ptr: scene) {
        delete shape_ptr;
    }
}

static void scene2() {
    std::vector<shape*> scene;
    //nsphere* sphere1 = new nsphere(point4(0.f, 0.f, -1.f, 0.4f), 0.5f, color(1.f, 0.647f, 0.f), 4);

    /*vec4 vx(-0.0971147f, -0.30548878f, -0.63968163f, -0.69860772f);
    vec4 vy(0.80357565f, -0.51840114f, 0.26369012f, -0.12646725f);
    vec4 vz(0.58114334f, 0.58723162f, -0.5408912f, 0.15769641f);
    vec4 vw(-0.08430502f, -0.54138331f, -0.47823806f, 0.68635641f); */

    std::unique_ptr<ncube> cube1 = std::make_unique<ncube>(point4(0.25f, 0.25f, 0.25f, 0.25f), color(1.f, 0.647f, 0.f));
    cube1->basis.rotate_xy(deg2rad(30.f));
    cube1->basis.rotate_yz(deg2rad(50.f));
    cube1->basis.rotate_zw(deg2rad(45.f));
    //cube1->basis.set_translation({ 0.2f, 0.2f, 0.f, 0.f });

    std::unique_ptr<nsphere> sphere2 = std::make_unique<nsphere>(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
    scene.push_back(cube1.get());
    scene.push_back(sphere2.get());

    std::vector<direction_light> lights;
    direction_light light1(vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
    lights.push_back(light1);

    camera cam{ scene, lights };
    cam.aspect_ratio = 16.f / 9.f;
    cam.image_width = 400;
    //cam.render_normals = true;
    cam.render();

    for (shape* shape_ptr : scene) {
        delete shape_ptr;
    }
}

static void scene3() {
    std::vector<shape*> scene;

    std::unique_ptr<cylinder> c1 = std::make_unique<cylinder>(point4(0.f, 0.f, 0.f, 0.f), point4(0.f, 0.5f, 0.f, 0.f), false, color(1.f, 0.647f, 0.f));
    std::unique_ptr<nsphere> sphere2 = std::make_unique<nsphere>(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);

    scene.push_back(c1.get());
    scene.push_back(sphere2.get());

    std::vector<direction_light> lights;
    direction_light light1(vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
    lights.push_back(light1);

    camera cam{ scene, lights };
    cam.aspect_ratio = 16.f / 9.f;
    cam.image_width = 400;
    //cam.render_normals = true;
    cam.render();

    for (shape* shape_ptr : scene) {
        delete shape_ptr;
    }
}

static void math_test() {
    mat4 m1 {
        -0.0971147f,  -0.30548878f, -0.63968163f, -0.69860772f,
        0.80357565f, -0.51840114f,  0.26369012f, -0.12646725f,
        0.58114334f,  0.58723162f, -0.5408912f,   0.15769641f,
        -0.08430502f, -0.54138331f, -0.47823806f,  0.68635641f
    };

    mat4 m2 = m1.transpose();
    mat4 m3 = matmul(m1, m2);
    std::cout << m3;

    mat4 m4{
        3, 4, 0, 2,
        3, 10, 7, 8,
        -5, 0, 1, 0,
        7, 4, -1, 2
    };
    vec4 v{ 6, 2, -4, 7 };
    std::cout << m4.vecmul(v) << std::endl;

    std::cout << m2;
}

static void rotation_test() {
    std::vector<shape*> scene;
    // USE UNIQUE PTRS
    std::unique_ptr<cylinder> c1 = std::make_unique<cylinder>(point4(0.f, 0.f, 0.f, 0.f), point4(0.f, 0.5f, 0.f, 0.f), false, color(1.f, 0.647f, 0.f));
    std::unique_ptr<nsphere> sphere2 = std::make_unique<nsphere>(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
    std::unique_ptr<nsphere> point = std::make_unique<nsphere>(point4(-1.f, 0.f, 0.f, 0.f), 0.03f, color(1.f, 0.f, 0.f), 4);
    c1->basis.rotate_xy_around_point(deg2rad(30.f), point4(-1.f, 0.f, 0.f, 0.f));

    scene.push_back(c1.get());
    scene.push_back(sphere2.get());
    scene.push_back(point.get());

    std::vector<direction_light> lights;
    direction_light light1(vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
    lights.push_back(light1);

    camera cam{ scene, lights };
    cam.aspect_ratio = 16.f / 9.f;
    cam.image_width = 400;
    //cam.render_normals = true;
    cam.render();
}

static void tesseract() {
    std::vector<shape*> scene;

    // Assuming side_length = 0.5
    float L = 0.5f; // This is the full side length, not half_side

    // Each coordinate is either 0.0f or L (0.5f)
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

    // Edges (pairs of vertex indices from the `vertices` array above)
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
    std::vector<std::unique_ptr<cylinder>> cylinders;

    for (int i = 0; i < 32; ++i) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        std::unique_ptr<cylinder> c = std::make_unique<cylinder>(vertices[idx1], vertices[idx2], true, edge_color);
        cylinders.push_back(std::move(c));

        scene.push_back(cylinders.back().get());
    }

    std::vector<direction_light> lights;
    direction_light light1(vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
    lights.push_back(light1);

    camera cam{ scene, lights };
    cam.aspect_ratio = 16.f / 9.f;
    cam.image_width = 400;
    //cam.render_normals = true;
    cam.render();
}

static float handle_inputs(const std::vector<shape*> scene, camera& cam) {
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
    std::vector<std::unique_ptr<shape>> unique_scene;
    std::vector<shape*> scene;

    for (int i = 0; i < 32; ++i) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        vec4 move(L / 2, L / 2, L / 2, L / 2);
        unique_scene.push_back(std::make_unique<cylinder>(vertices[idx1] - move, vertices[idx2] - move, true, edge_color));
        scene.push_back(unique_scene.back().get());
    }

    std::unique_ptr<cylinder> x_axis = std::make_unique<cylinder>(point4(0, 0, 0, 0), point4(L, 0, 0, 0), true, color(1, 0, 0), 0.01f);
    std::unique_ptr<cylinder> y_axis = std::make_unique<cylinder>(point4(0, 0, 0, 0), point4(0, L, 0, 0), true, color(0, 1, 0), 0.01f);
    std::unique_ptr<cylinder> z_axis = std::make_unique<cylinder>(point4(0, 0, 0, 0), point4(0, 0, L, 0), true, color(0, 0, 1), 0.01f);
    std::unique_ptr<cylinder> w_axis = std::make_unique<cylinder>(point4(0, 0, 0, 0), point4(0, 0, 0, L), true, color(0.73f, 0.33f, 0.827f), 0.01f);
    scene.push_back(x_axis.get());
    scene.push_back(y_axis.get());
    scene.push_back(z_axis.get());
    scene.push_back(w_axis.get());

    std::vector<direction_light> lights;
    direction_light light1(vec4::normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
    lights.push_back(light1);

    camera cam{ scene, lights };
    cam.aspect_ratio = 16.f / 9.f;
    cam.image_width = 640;
    //cam.render_normals = true;
    cam.initialize();

    // --- SETUP OPENGL TEXTURE ---
    GLuint render_texture;
    glGenTextures(1, &render_texture);
    glBindTexture(GL_TEXTURE_2D, render_texture);
    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Allocate memory for the texture on the GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_image_width, g_image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind

    int rt_latency = 0;
    int full_latency = 0;

    std::vector<std::thread> threads;
    std::barrier bar(N + 1);

    bool is_rendering = true; //std::atomic<bool> is_rendering = true;
    int first_row = 0;
    int row_range = std::ceil((float)g_image_height / N); // range: ceil(height / N)

    for (int i = 0; i < N; i++) {
        int last_row = std::min(first_row + row_range, g_image_height - 1);

        threads.push_back(std::thread([&cam, first_row, last_row, &is_rendering, &bar]() { 
            while (is_rendering) {
                cam.render_rt(first_row, last_row);
                bar.arrive_and_wait();
            }
        }));

        first_row += row_range;
    }

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

        // --- Render Target Window ---
        ImGui::Begin("Render Output");
        bool input_changed = handle_inputs(scene, cam);

        if (input_changed) {
            bar.arrive_and_wait(); // Check if our render thread has finished and provided new data
            glBindTexture(GL_TEXTURE_2D, render_texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_image_width, g_image_height, GL_RGB, GL_UNSIGNED_BYTE, g_image_data.data());
            glBindTexture(GL_TEXTURE_2D, 0); // Unbind
        }

        // Display the texture in an ImGui::Image widget
        ImGui::Image((void*)(intptr_t)render_texture, ImVec2(g_image_width, g_image_height));
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

    is_rendering = false;
    
    for (int i = 0; i < N; i++) {
        threads[i].join();
    }
    
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