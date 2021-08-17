CCOMP = g++

STD_CCOMP_FLAGS = -std=c++17 -Wall -pedantic -g
CCOMP_FLAGS = $(STD_CCOMP_FLAGS)
CLINK_FLAGS =
SHARED_LINK_FLAGS = $(CLINK_FLAGS) -shared

STD_LIBS = -lrt -lm
LIBS = $(STD_LIBS) -lsfml-graphics -lsfml-window -lsfml-system -limago

SRC_DIR = ./src
BLD_DIR = ./bld
BIN_DIR = ./bin

MKDIR = mkdir -p
RM = rm -rf

all: create clean graphics

graphics: graphics.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BIN_DIR)/$@ $(LIBS)

%.o: $(SRC_DIR)/%.cpp
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*