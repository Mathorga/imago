CCOMP = gcc

STD_CCOMP_FLAGS = -std=c++11 -Wall -pedantic -g -fopenmp
CCOMP_FLAGS = $(STD_CCOMP_FLAGS)
CLINK_FLAGS =

GTK_LIBS = `pkg-config --libs gtk+-2.0`
STD_LIBS = -lstdc++ -lrt -lgomp -lpthread -ldl -lm
LIBS = $(STD_LIBS)

SRC_DIR = ./src
BLD_DIR = ./bld
BIN_DIR = ./bin

MKDIR = mkdir -p
RM = rm -rf

all: default

default: create clean defaultExe

defaultExe: OortTest

%.o: $(SRC_DIR)/%.cpp
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

OortTest: OortTest.o \
		  Layer.o \
		  ConvolutionalLayer.o \
		  PoolingLayer.o \
		  Pooling2DLayer.o \
		  DenseLayer.o \
		  Model.o \
		  Knowledge.o \
		  Experience.o \
		  utils.o \
		  math.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BIN_DIR)/$@ $(LIBS)

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*
