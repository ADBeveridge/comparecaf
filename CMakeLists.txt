# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
project(mylib)

# Find dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(DEPS REQUIRED dlib>=19.0)

# Add subdirectory
add_subdirectory(src)
