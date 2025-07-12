#include <iostream>
#include <sstream>
#include <vector>
#include "color.h"
#include "ray.h"
#include "nsphere.h"
#include "camera.h"

color ray_color(const ray& r) {
    vec4 unit_direction = normalize(r.dir);
    float a = 0.5f * (unit_direction.y + 1.f);
    return (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
}

void scene1() {
    camera cam;
    cam.aspect_ratio = 16.f / 9.f;
    cam.image_width = 400;
    cam.focal_length = 1.f;

    std::vector<nsphere> scene;
    nsphere sphere1(point4(0.f, 0.f, -1.f, 0.f), 0.5f, 4);
    nsphere sphere2(point4(0.f, -100.5f, -1.f, 0.f), 100.f, 4);
    scene.push_back(sphere1);
    scene.push_back(sphere2);

    cam.render(scene);
}

int main() {
    scene1();

	return 0;
}