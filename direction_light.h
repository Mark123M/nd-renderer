#ifndef DIRECTION_LIGHT_H
#define DIRECTION_LIGHT_H

#include "vec4.h"
#include "color.h"
#include <cassert>
#include "util.h"

struct direction_light {
	vec4 dir;
	color col;

	direction_light(const vec4& dir0, const color& col0) : dir{ dir0 }, col{ col0 } {
		assert(std::abs(dir0.length_squared() - 1.f) <= TOL);
	}
};

#endif // !DIRECTION_LIGHT_H
