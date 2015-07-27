#pragma once

#include <algorithm>

using std::min;
using std::max;

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
