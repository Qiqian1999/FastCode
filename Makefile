# from stack overflow: ~/questions/21548464/how-to-write-a-makefile-to-compile-a-simple-c-program 
CC				= g++
CC_FLAGS 	= -mavx -mavx2 -mfma -O3 -std=c++1y
RM 				= rm -f

default: all 

all: usage


usage:
	$(CC) $(CC_FLAGS) -o use.o use.c Tensor.c Tensor.h Matrix.h Matrix.c Filters.c Filters.h Utility.c Utility.h


clean:
	rm -rf *.x *.S *.o
