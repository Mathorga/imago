CCOMP = gcc

STD_CCOMP_FLAGS = -std=c17 -Wall -pedantic -g
CCOMP_FLAGS = $(STD_CCOMP_FLAGS)
CLINK_FLAGS =
SHARED_LINK_FLAGS = $(CLINK_FLAGS) -shared

STD_LIBS = -lrt -lm
LIBS = $(STD_LIBS)

SRC_DIR = ./src
BLD_DIR = ./bld
BIN_DIR = ./bin

SYSTEM_INCLUDE_DIR = /usr/include
SYSTEM_LIB_DIR = /usr/lib

MKDIR = mkdir -p
RM = rm -rf

all: default

default: create clean test

# Installs the library files (headers and compiled) into the default system lookup folders.
install: create lib
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(BLD_DIR)/libimago.so $(SYSTEM_LIB_DIR)
#	sudo $(MKDIR) $(SYSTEM_LIB_DIR)/imago

uninstall: clean
	sudo $(RM) $(SYSTEM_INCLUDE_DIR)/imago
	sudo $(RM) $(SYSTEM_LIB_DIR)/libimago.so

# Builds all library files.
lib: libimago.so

libimago.a: standard.o
	ar -cvq $(BLD_DIR)/$@ $(patsubst %.o, $(BLD_DIR)/%.o, $^)

libimago.so: standard.o
	$(CCOMP) $(SHARED_LINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BLD_DIR)/$@

%.o: $(SRC_DIR)/%.c
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*
