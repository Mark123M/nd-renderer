#include <iostream>
#include <sstream>
#include <vector>
#include "color.h"
#include "ray.h"
#include "nsphere.h"
#include "camera.h"
#include "direction_light.h"

void scene1() {
    std::vector<nsphere> scene;
    nsphere sphere1(point4(0.f, 0.f, -1.f, 0.4f), 0.5f, color(1.f, 0.647f, 0.f), 4);
    nsphere sphere2(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
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
}

int main() {
    scene1();

	return 0;
}