# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(guise VERSION 1.0)

# Get the DLib library. 
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(DLIB dlib-1 REQUIRED)
endif()

# Add subdirectories
add_subdirectory(data)
add_subdirectory(src)
