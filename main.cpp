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

void scene1() {
    std::vector<shape*> scene;
    nsphere* sphere1 = new nsphere(point4(0.f, 0.f, -1.f, 0.4f), 0.5f, color(1.f, 0.647f, 0.f), 4);
    nsphere* sphere2 = new nsphere(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
    scene.push_back(sphere1);
    scene.push_back(sphere2);

    std::vector<direction_light> lights;
    direction_light light1(normalize(vec4(0.8f, 0.8f, 0.5f, 0.f)), color(1.0f, 1.0f, 0.9f));
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

void scene2() {
    std::vector<shape*> scene;
    //nsphere* sphere1 = new nsphere(point4(0.f, 0.f, -1.f, 0.4f), 0.5f, color(1.f, 0.647f, 0.f), 4);

    /*vec4 vx(-0.0971147f, -0.30548878f, -0.63968163f, -0.69860772f);
    vec4 vy(0.80357565f, -0.51840114f, 0.26369012f, -0.12646725f);
    vec4 vz(0.58114334f, 0.58723162f, -0.5408912f, 0.15769641f);
    vec4 vw(-0.08430502f, -0.54138331f, -0.47823806f, 0.68635641f); */

    ncube* cube1 = new ncube(point4(0.25f, 0.25f, 0.25f, 0.25f), color(1.f, 0.647f, 0.f));
    cube1->transform.rotate_xy(deg2rad(30.f));
    cube1->transform.rotate_yz(deg2rad(50.f));
    cube1->transform.rotate_zw(deg2rad(45.f));
    //cube1->transform.set_translation({ 0.2f, 0.2f, 0.f, 0.f });

    nsphere* sphere2 = new nsphere(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
    scene.push_back(cube1);
    scene.push_back(sphere2);

    std::vector<direction_light> lights;
    direction_light light1(normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
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

void scene3() {
    std::vector<shape*> scene;

    cylinder* c1 = new cylinder(point4(0.f, 0.f, 0.f, 0.f), point4(0.f, 0.5f, 0.f, 0.f), color(1.f, 0.647f, 0.f));
    nsphere* sphere2 = new nsphere(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);

    scene.push_back(c1);
    scene.push_back(sphere2);

    std::vector<direction_light> lights;
    direction_light light1(normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
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

void math_test() {
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

void rotation_test() {
    std::vector<shape*> scene;
    // USE UNIQUE PTRS
    std::unique_ptr<cylinder> c1 = std::make_unique<cylinder>(point4(0.f, 0.f, 0.f, 0.f), point4(0.f, 0.5f, 0.f, 0.f), color(1.f, 0.647f, 0.f));
    std::unique_ptr<nsphere> sphere2 = std::make_unique<nsphere>(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
    std::unique_ptr<nsphere> point = std::make_unique<nsphere>(point4(-1.f, 0.f, 0.f, 0.f), 0.03f, color(1.f, 0.f, 0.f), 4);
    c1->transform.rotate_xy_around_point(deg2rad(30.f), point4(-1.f, 0.f, 0.f, 0.f));

    scene.push_back(c1.get());
    scene.push_back(sphere2.get());
    scene.push_back(point.get());

    std::vector<direction_light> lights;
    direction_light light1(normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
    lights.push_back(light1);

    camera cam{ scene, lights };
    cam.aspect_ratio = 16.f / 9.f;
    cam.image_width = 400;
    //cam.render_normals = true;
    cam.render();
}

void skeleton() {
    std::vector<shape*> scene;

    // Assuming side_length = 0.5, so half_side = 0.25
    float hs = 0.25f; // half_side

    // Vertex Index (Binary representation of (WZYX) -> Coordinate signs)
    // Example: 0000 -> (-hs, -hs, -hs, -hs)
    //          1111 -> (+hs, +hs, +hs, +hs)

    // Vertices (as point4 or similar 4-component vector structure)
    point4 vertices[16] = {
        // ---- W= -hs Plane (analogous to 3D cube's 'bottom' face) ----
        // Z = -hs
        point4(-hs, -hs, -hs, -hs), // 0
        point4(hs, -hs, -hs, -hs), // 1
        point4(-hs,  hs, -hs, -hs), // 2
        point4(hs,  hs, -hs, -hs), // 3
        // Z = +hs
        point4(-hs, -hs,  hs, -hs), // 4
        point4(hs, -hs,  hs, -hs), // 5
        point4(-hs,  hs,  hs, -hs), // 6
        point4(hs,  hs,  hs, -hs), // 7

        // ---- W= +hs Plane (analogous to 3D cube's 'top' face) ----
        // Z = -hs
        point4(-hs, -hs, -hs,  hs), // 8
        point4(hs, -hs, -hs,  hs), // 9
        point4(-hs,  hs, -hs,  hs), // 10
        point4(hs,  hs, -hs,  hs), // 11
        // Z = +hs
        point4(-hs, -hs,  hs,  hs), // 12
        point4(hs, -hs,  hs,  hs), // 13
        point4(-hs,  hs,  hs,  hs), // 14
        point4(hs,  hs,  hs,  hs)  // 15
    };

    // Edges (pairs of vertex indices)
    int edges[32][2] = {
        // --- Edges within the W=-hs cube (indices 0-7) ---
        {0, 1}, // X-axis
        {0, 2}, // Y-axis
        {0, 4}, // Z-axis

        {1, 3},
        {1, 5},

        {2, 3},
        {2, 6},

        {3, 7},

        {4, 5},
        {4, 6},

        {5, 7},

        {6, 7},

        // --- Edges within the W=+hs cube (indices 8-15) ---
        {8, 9}, // X-axis
        {8, 10}, // Y-axis
        {8, 12}, // Z-axis

        {9, 11},
        {9, 13},

        {10, 11},
        {10, 14},

        {11, 15},

        {12, 13},
        {12, 14},

        {13, 15},

        {14, 15},

        // --- Edges connecting the W=-hs cube to the W=+hs cube (W-axis edges) ---
        {0, 8},
        {1, 9},
        {2, 10},
        {3, 11},
        {4, 12},
        {5, 13},
        {6, 14},
        {7, 15}
    };

    color edge_color(1.0f, 0.647f, 0.0f); // Orange
    float cylinder_radius = 0.02f; // Or whatever radius you choose

    for (int i = 0; i < 32; ++i) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        scene.push_back(new cylinder(vertices[idx1], vertices[idx2], edge_color, cylinder_radius));
    }

    // USE UNIQUE PTRS
   /* std::unique_ptr<cylinder> c1 = std::make_unique<cylinder>(point4(0.f, 0.f, 0.f, 0.f), point4(0.f, 0.5f, 0.f, 0.f), color(1.f, 0.647f, 0.f));
    std::unique_ptr<nsphere> sphere2 = std::make_unique<nsphere>(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
    std::unique_ptr<nsphere> point = std::make_unique<nsphere>(point4(-1.f, 0.f, 0.f, 0.f), 0.03f, color(1.f, 0.f, 0.f), 4);
    c1->transform.rotate_xy_around_point(deg2rad(30.f), point4(-1.f, 0.f, 0.f, 0.f));

    scene.push_back(c1.get());
    scene.push_back(sphere2.get());
    scene.push_back(point.get()); */

    std::vector<direction_light> lights;
    direction_light light1(normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
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

void tesseract() {
    std::vector<shape*> scene;

    // Assuming side_length = 0.5
    float L = 0.5f; // This is the full side length, not half_side

    // Vertices (as point4 or similar 4-component vector structure)
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

    for (int i = 0; i < 32; ++i) {
        int idx1 = edges[i][0];
        int idx2 = edges[i][1];

        scene.push_back(new cylinder(vertices[idx1], vertices[idx2], edge_color));
    }

    std::vector<direction_light> lights;
    direction_light light1(normalize(vec4(0.8f, 0.8f, 0.5f, 0.1f)), color(1.0f, 1.0f, 0.9f));
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

int main() {
    // scene1();
    // scene2();
    // math_test();
    // scene3();
    //rotation_test();
    //skeleton();

    tesseract();
	return 0;
}