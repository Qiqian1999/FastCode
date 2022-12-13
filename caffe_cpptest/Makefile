PROJECT_HOME = ./

CC = g++

BIN = ./inference

SRCS = ./inference.cpp

OBJS = $(SRCS:.cpp=.o)

CFLAGS = -O3 -DCPU_ONLY -std=c++11 -I/usr/include \
	-I /opt/caffe/include \
	-I /opt/caffe/build/include

LIBFLAGS = -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_videostab -lopencv_video -lopencv_highgui -lpthread -lboost_system -lglog -lgflags -lprotobuf \
	-L/opt/caffe/build/lib -lcaffe

all: $(BIN)

$(OBJS):

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

$(BIN): $(OBJS)
	$(CC) $^ $(LIBFLAGS) -o $@ 

clean:
	rm -f $(BIN) ./*.o
