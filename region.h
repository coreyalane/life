#pragma once

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

