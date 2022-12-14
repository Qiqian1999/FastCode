CXX=g++
CXXFLAGS=-g -O3 -std=c++11 -mfma -mavx2 -mavx -Wall -pedantic

BIN=run

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	    $(CXX) -o $(BIN) $^

%.o: %.c
	    $(CXX) $@ -c $<

clean:
	    rm -f *.o
		    rm $(BIN)
