#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <random>
#include <time.h>
#include <vector>

#include <cuda.h>
#include <GL/glut.h>

#include "entity.h"
#include "environment.h"
#include "key_press.h"
#include "point.h"
#include "region.h"
#include "region_grid.h"
#include "rng.h"
#include "util.h"

using std::abs;
using std::cin;
using std::cout;
using std::endl;
using std::list;
using std::max;
using std::min;
using std::queue;

const int BLOCK_SIZE = 128;

template<typename T>
__global__ void update_kernel(entity<T>* entities, region<T>* regions, const int max_entities, const T total_regions) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < max_entities) {
		entity<T>& ent = entities[idx];
		if(ent.active) {
			region<T>& current_region = regions[ent.region_index];
			ent.die(current_region, total_regions, max_entities);
		}
	}
}

template<typename T>
class game_state {
public:
	game_state():
		gen_count(0),
		entities(),
	 	environments(),
		next_environment_time(0),	
		regions(),
		entity_count(0),
		unused_entity_hint(0),
		draw_regions_enabled(false),
		draw_environments_enabled(false),
		draw_heat_tolerance_enabled(false),
		key_presses()
	{
		//entities = new entity<T>[MAX_ENTITIES]();
		cudaMallocManaged((void**)&entities, sizeof(entity<T>) * MAX_ENTITIES);
		cudaDeviceSynchronize();
		for(int i = 0; i<MAX_ENTITIES; ++i) {
			entities[i] = entity<T>();
		}
	}

	static const long MIN_ENVIRONMENT_LIFESPAN = 100;
	static const long MAX_ENVIRONMENT_LIFESPAN = 1000;
        static const int MAX_ENTITIES = 300000;
	constexpr static const T MOVE_STEP = 0.01;

	long gen_count;
	entity<T>* entities;
	list<environment<T>> environments; 
	long next_environment_time;
	region_grid<T> regions;
	int entity_count;
	int unused_entity_hint;
	bool draw_regions_enabled;
	bool draw_environments_enabled;
	bool draw_heat_tolerance_enabled;
	queue<key_press> key_presses;

	void time_step() {
		clock_t now = clock();
		handle_key_presses();
		if(entity_count <= 0) {
			int new_entity_index = create_entity();
			entities[new_entity_index].location.x = rng<T>::next_neg();
			entities[new_entity_index].location.y = rng<T>::next_neg();
			regions.add_entity(entities[new_entity_index]);
		}
		if(environments.size() < 10) {
			point<T> location(rng<T>::next_neg(), rng<T>::next_neg());
			T magnitude = rng<T>::next_neg() * 2.0;
			long expire_time = gen_count + (rng<T>::next() * util::seconds_to_generations(MAX_ENVIRONMENT_LIFESPAN)) + util::seconds_to_generations(MIN_ENVIRONMENT_LIFESPAN);
			if(environments.empty()) {
				magnitude = 2.0;
				expire_time = std::numeric_limits<long>::max();
			}
			environment<T> new_environment(location, magnitude, expire_time, gen_count);
			environments.push_back(new_environment);
		}
		for(auto& env : environments) {	
			if(gen_count > env.update_time) {
				regions.update_heat_level(env.location, env.get_update_magnitude());
			}
		}
		environments.erase(
			std::remove_if(environments.begin(), environments.end(), [&](const environment<T>& env) {
				if(gen_count > env.expire_time) {
					return true;
				}
				return false;
			}),
			environments.end());
		if(draw_regions_enabled) {
			draw_regions();
		}
		if(draw_environments_enabled) {
			for(const auto& env : environments) {
				draw(env);
			}
		}
		//pre cuda loop
		for(int i=0; i<MAX_ENTITIES; ++i) {
			entity<T>& ent = entities[i];
			if(ent.active) {
				procreate(ent);
				move(ent);
			}
		}
		cout << "pre cuda loop: " << clock() - now << endl;
		//cuda loop
		T total_regions = region_grid<T>::total_regions();
		if(false) {
			for(int i=0; i<MAX_ENTITIES; ++i) {
				entity<T>& ent = entities[i];
				if(ent.active) {
					region<T>& current_region = regions.get_region(ent);
					ent.die(current_region, total_regions, MAX_ENTITIES);
				}
			}
		} else {
			//int block_size = 128;
			const int n_blocks = MAX_ENTITIES/BLOCK_SIZE + (MAX_ENTITIES%BLOCK_SIZE == 0 ? 0:1);
			update_kernel <<< n_blocks, BLOCK_SIZE >>> (entities, regions.region_list, MAX_ENTITIES, total_regions);
			cudaDeviceSynchronize();
		}
		//post cuda loop
	        cout << "cuda loop: " << clock() - now << " " << entities[0].active << endl;	
		for(int i=0; i<MAX_ENTITIES; ++i) {
			entity<T>& ent = entities[i];
			if(ent.active) {
				if(!rng<T>::prob(ent.prob_to_live)) {
					ent.active = false;
					entity_count--;
					regions.remove_entity(ent);
				}
				draw(ent);
			}
		}
		
		cout << (clock() - now)/1000.0 << "  " << entity_count << endl;
		/**
		T avg_power = 0.0;
		int power_count = 0;
		for(int i=0; i<MAX_ENTITIES; ++i) {
			if(entities[i].active && (entities[i].power > .5)) {
				avg_power += entities[i].fertility;
				++power_count;
			}
		}
		**/
		++gen_count;
		cout << "post cuda loop: " << clock() - now << endl;
		//cout << avg_power / (T)power_count << endl;
		//cout << entities[0].location.x << " : " << entities[0].location.y << endl;
	}

	void handle_key_presses() {
		while(!key_presses.empty()) {
			key_press& p = key_presses.front();
			switch(p.key) {
				case GLUT_KEY_LEFT:
					draw_regions_enabled = !draw_regions_enabled;
					break;
				case GLUT_KEY_RIGHT:
					draw_environments_enabled = !draw_environments_enabled;
					break;
				case GLUT_KEY_UP:
					draw_heat_tolerance_enabled = !draw_heat_tolerance_enabled;
			}
			key_presses.pop();
		}
	}

	int create_entity() {
		while(true) {
			if(unused_entity_hint > MAX_ENTITIES) {
				unused_entity_hint = 0;
			}
			entity<T>& ent = entities[unused_entity_hint];
			if(!ent.active) {
				ent.active = true;
				break;
			}
			unused_entity_hint++;
		}
		entity_count++;
		return unused_entity_hint;
	}

	void copy_entity(entity<T>& entity_to_copy) {
		int new_ent_index = create_entity();
		entities[new_ent_index] = entity_to_copy;
		entities[new_ent_index].mutate();
		regions.add_entity(entities[new_ent_index]);
	}

	void procreate(entity<T>& entity_to_procreate) {
		T fertility = entity_to_procreate.fertility - pow(((T)entity_count/MAX_ENTITIES), 2.0);
		if(rng<T>::prob(fertility)) {
			copy_entity(entity_to_procreate);
		}
	}

	void move(entity<T>& entity_to_move) {
		if(rng<T>::prob(entity_to_move.mobility)) {
			point<T>& location = entity_to_move.location;
			T x_offset = rng<T>::next_neg() * MOVE_STEP;
			location.x += x_offset;
			util::bound_neg(location.x);
			T y_offset = rng<T>::next_neg() * MOVE_STEP;
			location.y += y_offset;
			util::bound_neg(location.y);
			regions.update_entity(entity_to_move);
		}
	}

	void draw(const entity<T>& entity_to_draw) {
		const T X_DRAW_OFFSET = .005 * entity_to_draw.power;
		const T Y_DRAW_OFFSET = .01 * entity_to_draw.power;

		T r = entity_to_draw.redness;
		T g = entity_to_draw.greenness;
		T b = entity_to_draw.blueness;
		if(draw_heat_tolerance_enabled) {
			const T& heat_tolerance = entity_to_draw.heat_tolerance;
			if(heat_tolerance > .5) {
				r = 1.0;
				g = 1 - (heat_tolerance - .5)*2.0;
				b = 1 - (heat_tolerance - .5)*2.0; 
			} else {
				r = (heat_tolerance * 2.0);
				g = (heat_tolerance * 2.0);
				b = 1.0; 
			}
		}

		draw(entity_to_draw.location, r, g, b,
			X_DRAW_OFFSET, Y_DRAW_OFFSET);
	}

	void draw(const environment<T>& env_to_draw) {
		T r = 0.0;
		T b = 0.0;
		if(env_to_draw.magnitude > 0) {
			r = 1.0;
		}
		else {
			b = 1.0;
		}
		draw_outline(env_to_draw.location, r, 0.0, b, .03 * env_to_draw.magnitude, .06 * env_to_draw.magnitude);	
	}

	void draw_regions() {
		T row_step = 2.0 / regions.REGION_ROWS;
		T col_step = 2.0 / regions.REGION_COLS;
		for(int i = 0; i<regions.total_regions(); ++i) {
			int row = i / regions.REGION_COLS;
			int col = i % regions.REGION_COLS;
			T y = (row_step * ((T)row + 0.5)) - 1;
			T x = (col_step * ((T)col + 0.5)) - 1;
			region<T>& region = regions.get_region(i);
			T r = 0.0;
			T g = 0.0;
			T b = 0.0;
			if(false) {
				region.heat_color(r, g, b);
			} else {
				region.type_color(r, g, b);
			}
			draw(point<T>(x, y), r, g, b, col_step/2, row_step/2);
		}
		//cin >> row_step;
	}

	inline static void draw(const point<T>& location, T r, T g, T b, T x_offset, T y_offset) {
		const T& x = location.x;
		const T& y = location.y;
		glBegin(GL_QUADS);
		glColor3f(r, g, b);
		glVertex2f(x - x_offset, y - y_offset);
		glVertex2f(x - x_offset, y + y_offset);
		glVertex2f(x + x_offset, y + y_offset);
		glVertex2f(x + x_offset, y - y_offset);
		glEnd();
	}

	inline static void draw_outline(const point<T>& location, T r, T g, T b, T x_offset, T y_offset) {
		const T& x = location.x;
		const T& y = location.y;
		glBegin(GL_LINE_STRIP);
		glColor3f(r, g, b);
		glVertex2f(x - x_offset, y - y_offset);
		glVertex2f(x - x_offset, y + y_offset);
		glVertex2f(x + x_offset, y + y_offset);
		glVertex2f(x + x_offset, y - y_offset);
		glVertex2f(x - x_offset, y - y_offset);
		glEnd();
	}


private:
};

