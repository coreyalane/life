#pragma once

#include <random>

using std::default_random_engine;
using std::uniform_real_distribution;

template<typename T>
class rng {
public:
        static const int RAND_LIST_LENGTH = 200000;
	static const int RAND_CYCLE_LENGTH = 1;

	static T* rand_list;
	static int rand_index;
	static int rand_cycle;

	static void init() {
		rand_list = new T[RAND_LIST_LENGTH]();
		reseed(4732985);
	}

	static void reseed(int seed) {
		rand_index = 0;
		rand_cycle = 0;
		//try { random_device dev; seed = dev(); } catch (void* e){}
		default_random_engine generator(seed);
		uniform_real_distribution<T> distribution(0,1.0);
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
				reseed(static_cast<int>(rand_list[0]));
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
