#pragma once

#include "point.h"

template<typename T>
class environment {
public:
	environment(const point<T>& location, T magnitude, long expire_time, long gen_count):
		location(location),
		magnitude(magnitude),
		expire_time(expire_time),
		update_time(0),
		update_period(0),
		update_count(0)
	{
		update_period = (expire_time - gen_count) / 20;
		update_time = gen_count;
	}

	T get_update_magnitude() {
		T update_magnitude = magnitude / 10.0;
		if(update_count >= 10) {
			update_magnitude = -update_magnitude;
		}
		update_time += update_period;
		++update_count;
		return update_magnitude;
	}

	point<T> location;
	T magnitude;
 	long expire_time;
	long update_time;
	long update_period;
	int update_count;
};

