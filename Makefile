CCOMP = gcc

STD_CCOMP_FLAGS = -std=c17 -Wall -pedantic -g
CCOMP_FLAGS = $(STD_CCOMP_FLAGS)
CLINK_FLAGS =

STD_LIBS = -lrt -lm
LIBS = $(STD_LIBS)

SRC_DIR = ./src
BLD_DIR = ./bld
BIN_DIR = ./bin

MKDIR = mkdir -p
RM = rm -rf

all: default

default: create clean defaultExe

defaultExe: ImagoTest

%.o: $(SRC_DIR)/%.c
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

ImagoTest: ImagoTest.o \
		   imago.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BIN_DIR)/$@ $(LIBS)

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*
