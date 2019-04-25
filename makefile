CPP=c++
CPP_OPTS=-std=c++11 -g

all:			swap

clean:
			rm swap

swap:	3kxswap_map.cpp
			$(CPP) $(CPP_OPTS) -o 3kxswap 3kxswap_map.cpp