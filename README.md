# Imago
<p align="center" width="100%">
    <img width="33%" src="/imago.png"> 
</p>
Spiking neural network implementation in pure C (and CUDA)

## How to compile
### Static library installation
Run `make` or `make all` to install all packages as a system-wide dynamic library (only works on linux at the moment).<br/>
Run `make standard` to install the CPU package as a system-wide dynamic library (only works on linux at the moment).<br/>
Run `make cuda` to install the CUDA GPU package as a system-wide dynamic library (only works on linux at the moment).<br/>
Run `make uninstall` to uninstall any previous installation.

WARNING: Every time you `make` the previous installation is overwritten.

## How to use
### Header files
Once the installation is complete you can include the library by `#include <imago/imago.h>` and directly use every function in the packages you compiled.
During linking you can specify `-limago` in order to link the compiled functions.

## TODO
Full CUDA support
