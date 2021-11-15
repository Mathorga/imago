CCOMP=g++
NVCOMP=nvcc

STD_CCOMP_FLAGS=-std=c++17 -Wall -pedantic -g
CCOMP_FLAGS=$(STD_CCOMP_FLAGS)
CLINK_FLAGS=-Wall
NVCOMP_FLAGS=
NVLINK_FLAGS=

STD_LIBS=-lrt -lm
GRAPHICS_LIBS=-lsfml-graphics -lsfml-window -lsfml-system
SPIKES_COUNT_LIBS=-lncurses
TIME_CUDA_LIBS=-lcuda -lcudart -lcublas -lcurand
IMAGO_LIBS=-limago

SRC_DIR=./src
BLD_DIR=./bld
BIN_DIR=./bin

# Adds BLD_DIR to object parameter names.
OBJS=$(patsubst %.o,$(BLD_DIR)/%.o,$@.o)

MKDIR=mkdir -p
RM=rm -rf

all: create clean standard cuda

standard: spikes_count graphics time_std
cuda: time_cuda

spikes_count:
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(SPIKES_COUNT_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated spikes count!\n\n"

graphics:
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(GRAPHICS_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated graphics!\n\n"

time_std:
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(GRAPHICS_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated standard timer!\n\n"

time_cuda:
	$(NVCOMP) $(NVCOMP_FLAGS) -c $(SRC_DIR)/$@.cu -o $(BLD_DIR)/$@.o
	$(NVCOMP) $(NVLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(TIME_CUDA_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated cuda timer!\n\n"

test:
	$(CCOMP) $(NVCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(NVLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated base test!\n\n"

# %.o: $(SRC_DIR)/%.cpp
# 	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*