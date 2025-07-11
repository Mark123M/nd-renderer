#include <iostream>
#include "color.h"
#include "ray.h"
#include <sstream>

inline std::tm localtime_xp(std::time_t timer) {
    std::tm bt{};
#if defined(__unix__)
    localtime_r(&timer, &bt);
#elif defined(_MSC_VER)
    localtime_s(&bt, &timer);
#else
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    bt = *std::localtime(&timer);
#endif
    return bt;
}

// "YYYY-MM-DD__HH-MM-SS"
inline std::string time_stamp(const std::string& fmt = "%F__%H-%M-%S") {
    auto bt = localtime_xp(std::time(0));
    char buf[64];
    return { buf, std::strftime(buf, sizeof(buf), fmt.c_str(), &bt) };
}

/*
 initialize();

        std::ofstream file{ "renders/render_" + time_stamp() + ".ppm", std::ios::app };

        clock_t t0 = clock();
        file << "P3\n" << image_width << " " << image_height << "\n255\n";

        for (int py = 0; py < image_height; py++) {
            std::clog << "\rScanlines remaining: " << (image_height - py) << "     " << std::flush;
            for (int px = 0; px < image_width; px++) {
                color pixel_col(0, 0, 0);

                for (int sy = 0; sy < samples_per_strata_side; sy++) {
                    for (int sx = 0; sx < samples_per_strata_side; sx++) {
                        //sampler.dimension = 0; // Reset sampler dimension for new ray
                        ray r = get_ray(py, px, sy, sx);
                        pixel_col += ray_color(r, world, lights);
                    }
                }

                write_color(file, pixel_col / samples_per_pixel);
            }
        }
        /*ray test = get_ray(500, 170);
        ray_color(test, world, lights);

int duration = clock() - t0;
std::clog << "\rDone " << duration << "ms                                                    \n";

std::stringstream ss;
ss << duration / 1000;
std::string duration_string = ss.str();
std::ofstream metadata{ "renders/render_" + time_stamp() + "_" + duration_string + "s.metadata.txt", std::ios::app };
metadata << "time elapsed: " << duration << " ms/" << duration / 1000.0 << " s/" << duration / 60000.0 << " m" << std::endl;
metadata << "width: " << image_width << " height: " << image_height << "  " << std::endl;
metadata << "samples: " << samples_per_pixel << " max depth: " << max_depth << std::endl;

file.close();
metadata.close();
*/

color ray_color(const ray& r) {
    vec4 unit_direction = normalize(r.dir);
    float a = 0.5f * (unit_direction.y + 1.f);
    return (1.f - a) * color(1.f, 1.f, 1.f) + a * color(0.5f, 0.7f, 1.f);
}

int main() {
    float aspect_ratio = 16.f / 9.f;
	int image_width = 400;
	int image_height = std::max(1.f, image_width / aspect_ratio);

    float focal_length = 1.f;
    float viewport_height = 2.f;
    float viewport_width = viewport_height * ((float)image_width / image_height);
    point4 camera_center = point4(0.f, 0.f, 0.f, 0.f);

    vec4 viewport_x = vec4(viewport_width, 0.f, 0.f, 0.f);
    vec4 viewport_y = vec4(0.f, -viewport_height, 0.f, 0.f);
    vec4 pixel_x = viewport_x / image_width;
    vec4 pixel_y = viewport_y / image_height;

    point4 viewport_upper_left = camera_center - vec4(0.f, 0.f, 0.f, focal_length) - viewport_x / 2 - viewport_y / 2;
    point4 viewport_pixel00 = viewport_upper_left + 0.5 * (pixel_x + pixel_y);

    std::ofstream file{ "renders/render_" + time_stamp() + ".ppm", std::ios::app };

    clock_t t0 = clock();
    file << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << "     " << std::flush;
        for (int i = 0; i < image_width; i++) {
            point4 pixel_center = viewport_pixel00 + (i * pixel_x) + (j * pixel_y);
            vec4 direction = pixel_center - camera_center;
            ray r(camera_center, direction);

            color color = ray_color(r);
            write_color(file, color);
        }
    }

    int duration = clock() - t0;
    std::clog << "\rDone " << duration << "ms                                                    \n";

    std::stringstream ss;
    ss << duration / 1000;
    std::string duration_string = ss.str();
    std::ofstream metadata{ "renders/render_" + time_stamp() + "_" + duration_string + "s.metadata.txt", std::ios::app };
    metadata << "time elapsed: " << duration << " ms/" << duration / 1000.0 << " s/" << duration / 60000.0 << " m" << std::endl;
    metadata << "width: " << image_width << " height: " << image_height << "  " << std::endl;
    //metadata << "samples: " << samples_per_pixel << " max depth: " << max_depth << std::endl;

    file.close();
    metadata.close();

	return 0;
}