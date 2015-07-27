#pragma once

#include <vector>

#include "region.h"

using std::vector;

template<typename T>
class region_grid {
public:
	region_grid():
		region_list(new region<T>[REGION_ROWS*REGION_COLS]())
	{
		setup_land();
	}

	static const int REGION_ROWS = 20;
	static const int REGION_COLS = 40;
	
	constexpr static const T total_regions() {
		return REGION_ROWS * REGION_COLS;
	}

	region<T>& get_region(int index) {
		return region_list[index];
	}

	region<T>& get_region(entity<T>& ent) {
		return region_list[ent.region_index];
	}

	void add_entity(entity<T>& ent) {
		ent.region_index = index(ent.location);
		add_entity_stats_to_region(ent);
	}

	void update_entity(entity<T>& ent) {
		T new_index = index(ent.location);
		if(new_index != ent.region_index) {
			remove_entity_stats_from_region(ent);
			ent.region_index = new_index;
			add_entity_stats_to_region(ent);
		}
	}

	void remove_entity(entity<T>& ent) {
		remove_entity_stats_from_region(ent);
	}

	//void add_environment(const environment<T>& env) {
	//	update_heat_level(env.location, env.magnitude);
	//}

	//void remove_environment(const environment<T>& env) {
	//	update_heat_level(env.location, -env.magnitude);
	//}

	void update_heat_level(const point<T>& location, T magnitude) {
		int source_row = location_to_bin(location.y, REGION_ROWS); 
		int source_col = location_to_bin(location.x, REGION_COLS);
		static const int max_spread_radius = 20;
		for(int spread_radius = 0; spread_radius<=max_spread_radius; ++spread_radius) {
			vector<int> bins_in_radius = get_bins_in_radius(source_row, source_col, spread_radius);
			for(int bin : bins_in_radius) {
				region_list[bin].heat_level += (magnitude / static_cast<T>(max_spread_radius));
			}
		}
	}

private:
	region<T>* region_list;

	void setup_land() {
		for(int i=REGION_ROWS*1/4; i<=REGION_ROWS*3/4; ++i) {
			for(int j=REGION_COLS*1/5; j<REGION_COLS*2/5; ++j) {
				int region_index = index(i, j);
				region_list[region_index].type = region<T>::LAND;
				if(i == REGION_ROWS*3/4) {
					region_list[region_index].type = region<T>::SWAMP;
				}
			}
		}
		for(int i=REGION_ROWS*1/8; i<REGION_ROWS*2/8; ++i) {
			for(int j=REGION_COLS*4/7; j<REGION_COLS*5/7; ++j) {
				int region_index = index(i, j);
				region_list[region_index].type = region<T>::LAND;
			}
		}
		for(int i=REGION_ROWS*3/8; i<REGION_ROWS*4/8; ++i) {
			for(int j=REGION_COLS*5/7; j<=REGION_COLS*6/7; ++j) {
				int region_index = index(i, j);
				region_list[region_index].type = region<T>::LAND;
				if(j == REGION_COLS*6/7) {
					region_list[region_index].type = region<T>::SWAMP;
				}
			}
		}
		for(int i=REGION_ROWS*5/8; i<REGION_ROWS*6/8; ++i) {
			for(int j=REGION_COLS*4/7; j<REGION_COLS*5/7; ++j) {
				int region_index = index(i, j);
				region_list[region_index].type = region<T>::SWAMP;
			}
		}
	}

	int index(int row, int col) const {
		return (row * REGION_COLS) + col;
	}

	int index(const point<T>& p) const {
		int row = location_to_bin(p.y, REGION_ROWS);
		int col = location_to_bin(p.x, REGION_COLS);
		return index(row, col);
	}

	int location_to_bin(T location, int divisions) const {
		T zero_to_one = (location+1.0)/2.0;
		T zero_to_max = zero_to_one * divisions;
		int bin = static_cast<int>(zero_to_max);
		if(bin == divisions) {
			--bin;
		}
		return bin;
	}

	vector<int> get_bins_in_radius(int start_row, int start_col, int radius) {
		vector<int> bins_in_radius;
		for(int i=start_row-radius; i<=start_row+radius; ++i) {
			if((i >= 0) && (i < REGION_ROWS)) {
				for(int j=start_col-radius; j<=start_col+radius; ++j) {
					if((j >= 0) && (j < REGION_COLS)) {
						bins_in_radius.push_back(index(i, j));
					}	
				}
			} 
		}
		return bins_in_radius;
	}

	void add_entity_stats_to_region(const entity<T>& ent) {
		region_list[ent.region_index].count++;
		region_list[ent.region_index].power += ent.power;
	}

	void remove_entity_stats_from_region(const entity<T>& ent) {
		region_list[ent.region_index].count--;
		region_list[ent.region_index].power -= ent.power;
	}
};

