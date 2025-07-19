#include <iostream>
#include <sstream>
#include <vector>
#include "color.h"
#include "ray.h"
#include "nsphere.h"
#include "ncube.h"
#include "camera.h"
#include "direction_light.h"
#include "mat4.h"

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
    cube1->transform.rotate_xz(deg2rad(30.f));
    cube1->transform.rotate_yw(deg2rad(50.f));
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

int main() {
    // scene1();
    scene2();
    //math_test();
	return 0;
}