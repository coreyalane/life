#pragma once

#include "point.h"
#include "rng.h"
#include "util.h"

template<typename T>
class entity {
public:
	entity():
		active(false),
		location(0.0, 0.0),
		region_index(-1),
		fertility(0.5),
		mobility(0.1),
		redness(0.5),
		greenness(0.5),
		blueness(0.5),
		heat_tolerance(0.5),
		power(0.4),
		aqua_terra(0.0)
	{}

	bool active;
	point<T> location;
	int region_index;
	T fertility;
	T mobility;
	T redness;
	T greenness;
	T blueness;
	T heat_tolerance;
	T power;
	T aqua_terra;

	void mutate() {
		m(fertility);
		m(mobility);
		m(redness);
		m(greenness);
		m(blueness);
		m(heat_tolerance);
		m(power);
		m(aqua_terra);
	}

	T energy() {
		return (fertility + mobility + abs((heat_tolerance - 0.5) * 2.0) + power) / 4.0;
	}

private:
	constexpr static const T MUTATE_RATE = 0.15; 
	
	static void m(T& property) {
		if(rng<T>::prob(MUTATE_RATE)) {
			property *= (1 + (rng<T>::next_neg() * MUTATE_RATE));
			util::bound(property, .05, .95);
		}
	}
};

