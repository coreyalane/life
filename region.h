#pragma once

template<typename T>
class region {
public:
	region():
		type(WATER),
		count(0),
		heat_level(0.0),
		power(0.0)
	{}
	
	enum region_type {
		LAND,
		SWAMP,
		WATER
	};

	region_type type;
	int count;
	T heat_level;
	T power;

	__device__ __host__ T aqua_terra() const {
		switch(type) {
			case LAND:
				return 0.95;
			case SWAMP:
				return 0.5;
			case WATER:
				return 0.05;
		}
		return 0.05;
	}

	void heat_color(T& r, T& g, T& b) const {
		if(heat_level > 0) {
			r = min(heat_level, 1.0);
		}
		else {
			b = max(-heat_level, 0.0);
		}	
	}

	void type_color(T& r, T& g, T& b) const {
		switch(type) {
			case LAND:
				r = 0.0;
				g = 0.1;
				b = 0.0;
				break;
			case SWAMP:
				r = 0.0;
				g = 0.1;
				b = 0.1;
				break;
			case WATER:
				r = 0.0;
				g = 0.0;
				b = 0.1;
		}
	}
};

