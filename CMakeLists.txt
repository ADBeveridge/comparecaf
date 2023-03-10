# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(guise)

include(GNUInstallDirs)
include(FetchContent)

# Suppress -Wdeprecated error messages. This comes from DLib's use of std::iterator.
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
endif()

# Get DLib source, and make sure that it is being compiled with position independent code. 
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(DLIB dlib-1 REQUIRED)
endif()

# Add subdirectory
add_subdirectory(data)
add_subdirectory(src)
