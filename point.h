#pragma once

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
