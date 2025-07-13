#ifndef NSPHERE_H
#define NSPHERE_H

#include "vec4.h"
#include "color.h"


class nsphere {
	point4 center;
	//color albedo;
	float radius;
	int n;
public:
	nsphere(const point4& center, float radius, int n = 4): center{center}, radius{radius}, n{n} {}

	float sdf(const point4& p) {
		return (center - p).length() - radius;
	}
};

#endif