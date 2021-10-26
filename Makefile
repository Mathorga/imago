CCOMP = gcc
NVCOMP = nvcc

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

# Installs the library files (headers and compiled) into the default system lookup folders.
all: create lib
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(BLD_DIR)/libimago.so $(SYSTEM_LIB_DIR)
	@printf "\nInstallation complete!\n"

standard: create stdlib
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(BLD_DIR)/libimago.so $(SYSTEM_LIB_DIR)
	@printf "\nInstallation complete!\n"

cuda: create cudalib
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/imago
	sudo cp $(BLD_DIR)/libimago.so $(SYSTEM_LIB_DIR)
	@printf "\nInstallation complete!\n"

uninstall: clean
	sudo $(RM) $(SYSTEM_INCLUDE_DIR)/imago
	sudo $(RM) $(SYSTEM_LIB_DIR)/libimago.so
	@printf "\nSuccessfully uninstalled.\n"



# Unused static lib.
static: imago_std.o
	ar -cvq $(BLD_DIR)/libimago.a $(patsubst %.o, $(BLD_DIR)/%.o, $^)


# Builds all library files.
stdlib: imago_std.o
	$(CCOMP) $(SHARED_LINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BLD_DIR)/libimago.so

cudalib: imago_cuda.o
	$(CCOMP) $(SHARED_LINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BLD_DIR)/libimago.so

lib: imago_std.o imago_cuda.o
	$(CCOMP) $(SHARED_LINK_FLAGS) $(patsubst %.o, $(BLD_DIR)/%.o, $^) -o $(BLD_DIR)/libimago.so



# Builds object files from source.
%.o: $(SRC_DIR)/%.c
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

%.o: $(SRC_DIR)/%.cu
	$(NVCOMP) --compiler-options '-fPIC' -c $^ -o $(BLD_DIR)/$@



# Creates temporary working directories.
create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

# Removes temporary working directories.
clean:
	$(RM) $(BLD_DIR)/*
	$(RM) $(BIN_DIR)/*
