#pragma once

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

