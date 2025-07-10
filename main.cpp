#include <iostream>
#include "color.h"
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

int main() {
	int image_width = 256;
	int image_height = 256;

    std::ofstream file{ "renders/render_" + time_stamp() + ".ppm", std::ios::app };

    clock_t t0 = clock();
    file << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << "     " << std::flush;
        for (int i = 0; i < image_width; i++) {
            float r = (float)i / (image_width - 1);
            float g = (float)j / (image_height - 1);
            float b = 0.f;

            write_color(file, {r, g, b});
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