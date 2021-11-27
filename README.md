# Imago
<p align="center" width="100%">
    <img width="33%" src="/imago.png"> 
</p>
Spiking neural network implementation in pure C (and CUDA)

## Before you compile
Run the dependencies installation script: **install_deps.sh**.<br/>
This will make sure you are ready to launch the following comands.

## Shared library installation (Linux)
Run `make` or `make all` to install all packages in a system-wide dynamic library.<br/>

### Standard
Run `make standard` to install the CPU package as a system-wide dynamic library.<br/>

### CUDA
Run `make cuda` to install the CUDA GPU package as a system-wide dynamic library.<br/>
You can specify your card's compute architecture by setting the CUDA_ARCH variable at compilation time. e.g. `make cuda CUDA_ARCH=sm_35` if your card's compute capability is 3.5. This is useful when dealing with old GPUs that don't support the latest compilation strategies.<br/>

If you're having trouble running your program give this a shot.

### Uninstall
Run `make uninstall` to uninstall any previous installation.

WARNING: Every time you `make` a new package the previous installation is overwritten.

## How to use
### Header files
Once the installation is complete you can include the library by `#include <imago/imago.h>` and directly use every function in the packages you compiled.<br/>

### Linking
During linking you can specify `-limago` in order to link the compiled functions.

## Samples
You can find useful information in the [samples](samples) directory.

## TODO
Full CUDA support
