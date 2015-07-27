#include <algorithm>
#include <iostream>
#include <random>
#include <time.h>
#include <vector>
#include <limits>
#include <list>
#include <queue>

#include <GL/glut.h>

using namespace std;

typedef double fp_type;

namespace util {
template<typename T>
void bound(T& value, T low, T high) {
	value = max(low, min(high, value));
}

template<typename T>
void bound(T& value) {
	bound(value, 0.0, 1.0);
}

template<typename T>
void bound_neg(T& value) {
	bound(value, -1.0, 1.0);
}

long seconds_to_generations(long seconds) {
	return seconds * 30;
}
} //util

template<typename T>
class rng {
public:
        static const int RAND_LIST_LENGTH = 200000;
	static const int RAND_CYCLE_LENGTH = 1;

	static T* rand_list;
	static int rand_index;
	static int rand_cycle;

	static void init() {
		init(4732985);
	}

	static void init(int seed) {
		rand_list = new T[RAND_LIST_LENGTH]();
		rand_index = 0;
		rand_cycle = 0;
		//try { random_device dev; seed = dev(); } catch (void* e){}
		default_random_engine generator(seed);
		uniform_real_distribution<fp_type> distribution(0,1.0);
		for(int i=0; i<RAND_LIST_LENGTH; ++i) {
			rand_list[i] = distribution(generator);
		}
	}

	static T next() {
		T rand = rand_list[rand_index];
		++rand_index;
		if(rand_index > RAND_LIST_LENGTH) {
			rand_index = 0;
			++rand_cycle;
			if(rand_cycle > RAND_CYCLE_LENGTH) {
				init(static_cast<int>(rand_list[0]));
				rand_cycle = 0;
			}
		}
		return rand;
	}

	static bool prob(T p) {
		if(next() < p) {
			return true;
		}
		return false;
	}

	static T next_neg() {
		T rand = next();
		if(prob(.5)) {
			rand = -rand;
		}
		return rand;
	}
};
template<>
fp_type* rng<fp_type>::rand_list = nullptr;
template<>
int rng<fp_type>::rand_index = 0;
template<>
int rng<fp_type>::rand_cycle = 0;

template<typename T>
class point {
public:
	point(T x, T y):
		x(x),
		y(y)
	{}

	T x;
	T y;
};

template<typename T>
class entity {
public:
	entity():
		active(false),
		location(0.0, 0.0),
		region_index(-1),
		fertility(0.2),
		mobility(0.1),
		redness(0.5),
		greenness(0.5),
		blueness(0.5),
		heat_tolerance(0.5),
		power(0.4)
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

	void mutate() {
		m(fertility);
		m(mobility);
		m(redness);
		m(greenness);
		m(blueness);
		m(heat_tolerance);
		m(power);
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

template<typename T>
class region {
public:
	region():
		count(0),
		heat_level(0.0),
		power(0.0)
	{}

	int count;
	T heat_level;
	T power;
};

template<typename T>
class region_grid {
public:
	region_grid():
		region_list(new region<T>[REGION_ROWS*REGION_COLS]())
	{}

	static const int REGION_ROWS = 40;
	static const int REGION_COLS = 80;
	
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

class key_press {
public:
	key_press(int key, int x, int y):
		key(key),
		x(x),
		y(y)
	{}
	int key;
	int x;
	int y;
};

template<typename T>
class game_state {
public:
	game_state():
		gen_count(0),
		entities(new entity<T>[MAX_ENTITIES]()),
	 	environments(),
		next_environment_time(0),	
		regions(),
		entity_count(0),
		unused_entity_hint(0),
		draw_regions_enabled(false),
		draw_environments_enabled(false),
		draw_heat_tolerance_enabled(false),
		key_presses()
	{}

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
		for(int i=0; i<MAX_ENTITIES; ++i) {
			entity<T>& ent = entities[i];
			if(ent.active) {
				procreate(ent);
				move(ent);
				die(ent);
				draw(ent);
			}
		}
		if(draw_environments_enabled) {
			for(const auto& env : environments) {
				draw(env);
			}
		}
		if(draw_regions_enabled) {
			draw_regions();
		}
		cout << (clock() - now)/1000.0 << "  " << entity_count << endl;
		T avg_power = 0.0;
		int power_count = 0;
		for(int i=0; i<MAX_ENTITIES; ++i) {
			if(entities[i].active && (entities[i].power > .5)) {
				avg_power += entities[i].fertility;
				++power_count;
			}
		}
		++gen_count;
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
		T fertility = entity_to_procreate.fertility - ((T)entity_count/MAX_ENTITIES);
		if(rng<T>::prob(fertility)) {
			copy_entity(entity_to_procreate);
		}
	}

	void die(entity<T>& entity_to_die) {
		T prob_to_live = 1.0;
		//cout << "base prob to live: " << prob_to_live << endl;

		//region
		region<T>& current_region = regions.get_region(entity_to_die);
		const T region_count = static_cast<T>(current_region.count);
		static const T total_regions = region_grid<T>::total_regions();
		static const T average_count_per_region = static_cast<T>(MAX_ENTITIES) / total_regions;
		const T& region_heat_level = current_region.heat_level;
		const T& region_power = current_region.power;
		const T& average_region_power = region_power / region_count;

		//overpopulation
		static const T upper_population_limit = average_count_per_region * 3.0;
		T prob_overcrowding = min(region_count / upper_population_limit, 1.0) * (1 - entity_to_die.power);  
		prob_to_live *= (1.0 - prob_overcrowding);
		//cout << "prob overcrowd: " << prob_overcrowding << endl;

		//heat
		T scaled_heat_tolerance = (entity_to_die.heat_tolerance - 0.5) * 1.0;
		T prob_overheating = min(abs(scaled_heat_tolerance - region_heat_level), 1.0);
		prob_to_live *= (1.0 - prob_overheating);
 		//cout << "prob overheat: " << prob_overheating << endl; 

		//starving
		T relative_power = entity_to_die.power / average_region_power; // [0, inf]
		T capped_relative_power = min(relative_power, 1.0); // [0, 1]
		T prob_starving = capped_relative_power * entity_to_die.power / 2.0;
		prob_to_live *= (1.0 - prob_starving); 
		//cout << "prob starve: " << prob_starving << endl;

		//edge
		prob_to_live *= 1 - pow(entity_to_die.location.x, 10.0);
		prob_to_live *= 1 - pow(entity_to_die.location.y, 10.0);

		//cout << "final prob to live: " << prob_to_live << endl;

		//calc death
		if(!rng<T>::prob(prob_to_live)) {
			entity_to_die.active = false;
			entity_count--;
			regions.remove_entity(entity_to_die);
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
		draw(env_to_draw.location, r, 0.0, b, .03 * env_to_draw.magnitude, .06 * env_to_draw.magnitude);	
	}

	void draw_regions() {
		T row_step = 2.0 / regions.REGION_ROWS;
		T col_step = 2.0 / regions.REGION_COLS;
		for(int i = 0; i<regions.total_regions(); ++i) {
			int row = i / regions.REGION_COLS;
			int col = i % regions.REGION_COLS;
			T y = (row_step * (row + 0.5)) - 1;
			T x = (col_step * (col + 0.5)) - 1;
			T heat_level = regions.get_region(i).heat_level;
			T r = 0.0;
			T b = 0.0;
			if(heat_level > 0) {
				r = min(heat_level, 1.0);
			}
			else {
				b = max(-heat_level, 0.0);
			}
			draw(point<T>(x, y), r, 0.0, b, row_step, col_step);
		}
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

private:
};

game_state<fp_type>* gs = nullptr;

void draw() {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	gs->time_step();
	glutSwapBuffers();
	glFinish();
	glutPostRedisplay();
}

void keyboard(int key, int x, int y) {
	gs->key_presses.emplace(key, x, y);
}

void setupGlut(int argc, char** argv) {
	int WIDTH = 1440;
	int HEIGHT = 720;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
	glutInitWindowPosition(50, 25);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Hello OpenGL");
	glutDisplayFunc(draw);
	glutSpecialFunc(keyboard);

	glEnable(GL_TEXTURE_2D);
	glShadeModel(GL_SMOOTH);        
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);                
        glClearDepth(1);                                       
        //glEnable(GL_BLEND);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glViewport(0,0,WIDTH,HEIGHT);
	glMatrixMode(GL_MODELVIEW);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1, 1, -1, 1, 1, -1);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv) {
	cout << "Hello World" << endl;
	
	rng<fp_type>::init();
	gs = new game_state<fp_type>();
	setupGlut(argc, argv);

	glutMainLoop();

	return 0;
}
