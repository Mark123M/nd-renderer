#include <iostream>
#include <sstream>
#include <vector>
#include "color.h"
#include "ray.h"
#include "nsphere.h"
#include "camera.h"

void scene1() {
    std::vector<nsphere> scene;
    nsphere sphere1(point4(0.f, 0.f, 0.f, -1.f), 0.5f, 4);
    nsphere sphere2(point4(0.f, -100.5f, 0.f, -1.f), 100.f, 4);
    scene.push_back(sphere1);
    scene.push_back(sphere2);

    camera cam{ 16.f / 9.f, 400, 1.f, scene };
    cam.render();
}

int main() {
    scene1();

	return 0;
}