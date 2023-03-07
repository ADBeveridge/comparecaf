# Guise
A library to compare faces from images

Guise, given two images (that may be the same), will scan each image to see if it contains a face that the other image specified has.

## Installing for Usage
You will need:
 - G++ (or whatever compiler you use for C++)
 - CMake
 - Dlib
 - FlexiBLAS
 - Sqlite3
 - LibX11
 - LibJPEG
 - LibPNG
 - LibZ
 
 ### Fedora
 ```
 sudo dnf install dlib-devel flexiblas-devel cmake sqlite-devel libX11-devel libjpeg-devel libpng-devel zlib-devel g++
 ```
Once you have all your dependencies, download Guise, and extract it if needed (not if cloned via Git). Then follow the basic CMake compilation process.
### Fedora / Debian
``` 
git clone https://github.com/adbeveridge/guise.git
cd guise
mkdir build
cd build
cmake ..
make
sudo make install
```
