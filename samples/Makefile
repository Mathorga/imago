CCOMP = g++

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

all: create clean samples

samples: spikes_count graphics

spikes_count: spikes_count.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BIN_DIR)/$@ $(STD_LIBS) $(SPIKES_COUNT_LIBS) $(IMAGO_LIBS)

graphics: graphics.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BIN_DIR)/$@ $(STD_LIBS) $(GRAPHICS_LIBS) $(IMAGO_LIBS)

%.o: $(SRC_DIR)/%.cpp
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*