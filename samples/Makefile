CCOMP = g++
NVCOMP = nvcc

STD_CCOMP_FLAGS = -std=c++17 -Wall -pedantic -g
CCOMP_FLAGS = $(STD_CCOMP_FLAGS)
CLINK_FLAGS =
SHARED_LINK_FLAGS = $(CLINK_FLAGS) -shared

STD_LIBS = -lrt -lm
GRAPHICS_LIBS = -lsfml-graphics -lsfml-window -lsfml-system
SPIKES_COUNT_LIBS = -lncurses
IMAGO_LIBS = -limago

SRC_DIR = ./src
BLD_DIR = ./bld
BIN_DIR = ./bin

MKDIR = mkdir -p
RM = rm -rf

all: create clean standard cuda

standard: spikes_count graphics time_std
cuda: time_cuda

spikes_count:
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $@.o) -o $(BIN_DIR)/$@ $(STD_LIBS) $(SPIKES_COUNT_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated spikes count!\n\n"

graphics:
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $@.o) -o $(BIN_DIR)/$@ $(STD_LIBS) $(GRAPHICS_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated graphics!\n\n"

time_std:
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $@.o) -o $(BIN_DIR)/$@ $(STD_LIBS) $(GRAPHICS_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated standard timer!\n\n"

time_cuda:
	$(NVCOMP) -c $(SRC_DIR)/$@.cu -o $(BLD_DIR)/$@.o
	$(NVCOMP) $(patsubst %.o, $(BLD_DIR)/%.o, $@.o) -o $(BIN_DIR)/$@ $(STD_LIBS) $(GRAPHICS_LIBS) $(IMAGO_LIBS)
	@printf "\nCreated cuda timer!\n\n"

# %.o: $(SRC_DIR)/%.cpp
# 	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*