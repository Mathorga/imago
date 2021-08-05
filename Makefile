CCOMP = gcc

STD_CCOMP_FLAGS = -std=c17 -Wall -pedantic -g
CCOMP_FLAGS = $(STD_CCOMP_FLAGS)
CLINK_FLAGS =

STD_LIBS = -lrt -lm
LIBS = $(STD_LIBS)

SRC_DIR = ./src
BLD_DIR = ./bld
BIN_DIR = ./bin

SYSTEM_INCLUDE_DIR = /usr/include
SYSTEM_LIB_DIR = /usr/local/lib

MKDIR = mkdir -p
RM = rm -rf

all: default

default: create clean test

test: imago_test

# Installs the library files (headers and compiled) into the default system lookup folders.
install: libimago.a
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/imago
	sudo $(MKDIR) $(SYSTEM_LIB_DIR)/imago
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(BLD_DIR)/libimago.a $(SYSTEM_LIB_DIR)

uninstall: clean
	sudo $(RM) $(SYSTEM_INCLUDE_DIR)/imago
	sudo $(RM) $(SYSTEM_LIB_DIR)/imago

# Builds all library files.
lib: libimago.a

libimago.a: standard.o
	ar -cvq $(BLD_DIR)/$@ $(patsubst %.o, $(BLD_DIR)/%.o, $^)

%.o: $(SRC_DIR)/%.c
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

imago_test: imago_test.o \
			standard.o
	$(CCOMP) $(CLINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BIN_DIR)/$@ $(LIBS)

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*
