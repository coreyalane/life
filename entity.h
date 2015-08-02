#pragma once

#include "point.h"
#include "region.h"
#include "rng.h"
#include "util.h"

template<typename T>
class entity {
public:
	entity():
		active(false),
		location(0.0, 0.0),
		region_index(-1),
		prob_to_live(1.0),
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
	T prob_to_live;
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

	__device__ __host__ void die(region<T>& current_region, const T total_regions, const int MAX_ENTITIES) {
		prob_to_live = 1.0;
		//cout << "base prob to live: " << prob_to_live << endl;

		//region
		const T region_count = static_cast<T>(current_region.count);
		const T average_count_per_region = static_cast<T>(MAX_ENTITIES) / total_regions;
		const T& region_heat_level = current_region.heat_level;
		const T& region_power = current_region.power;
		const T& average_region_power = region_power / region_count;
		const T& region_aqua_terra = current_region.aqua_terra();

		//overpopulation
		const T upper_population_limit = average_count_per_region * 3.0;
		T prob_overcrowding = min(region_count / upper_population_limit, 1.0) * (1 - power);  
		prob_to_live *= (1.0 - prob_overcrowding);
		//cout << "prob overcrowd: " << prob_overcrowding << endl;

		//heat
		T scaled_heat_tolerance = (heat_tolerance - 0.5) * 1.0;
		T prob_overheating = min(abs(scaled_heat_tolerance - region_heat_level), 1.0);
		prob_to_live *= (1.0 - prob_overheating);
 		//cout << "prob overheat: " << prob_overheating << endl; 

		//aqua_terra
		T prob_suffocate = abs(region_aqua_terra - aqua_terra);
		prob_to_live *= (1.0 - prob_suffocate);
		//cout << prob_suffocate << endl;

		//starving
		T relative_power = power / average_region_power; // [0, inf]
		T capped_relative_power = min(relative_power, 1.0); // [0, 1]
		T prob_starving = capped_relative_power * power / 2.0;
		prob_to_live *= (1.0 - prob_starving); 
		//cout << "prob starve: " << prob_starving << endl;

		//edge
		prob_to_live *= 1 - pow(location.x, 10.0);
		prob_to_live *= 1 - pow(location.y, 10.0);

		//cout << "final prob to live: " << prob_to_live << endl;

		//return prob_to_live;
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
